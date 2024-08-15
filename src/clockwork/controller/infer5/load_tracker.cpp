#include "clockwork/controller/infer5/load_tracker.h"
#include "clockwork/util.h"
#include "dmlc/logging.h"

namespace clockwork {
namespace scheduler {
namespace infer5 {

ModelLoadTracker* LoadTracker::newModelTracker(int model_id) {
    return new ModelLoadTracker(capacity, model_id, gpus.size());
}
   
ModelLoadTracker::ModelLoadTracker(int64_t capacity, int model_id, int n_gpus) : 
        capacity(capacity),
        model_id(model_id), 
        touched(n_gpus) {
    for (int i = 0; i < touched.size(); i++) {
        touched[i] = false;
    }
}

/**
* Calculates and returns a LoadTracker::Demand object containing information about the execution time of loading/infer.
*
* @param size The size of the request.
* @param start_exec_by The time by which the execution should start.
* @param start_loadweights_by The time by which the weights loading should start.
*
* @return A LoadTracker::Demand object containing information about the execution time of loading/infer.
*
* @throws None.
*/
LoadTracker::Demand ModelLoadTracker::addRequest(int64_t size, uint64_t start_exec_by, uint64_t start_loadweights_by) {
    LoadTracker::Demand demand;
    demand.exec_size = (size * capacity) / start_exec_by;
    demand.loadweights_size = (size * capacity) / start_loadweights_by;
    demand.model_id = model_id;

    this->start_loadweights_by = start_loadweights_by;
    new_exec += demand.exec_size;
    new_loadweights += demand.loadweights_size;

    return demand;
}

void ModelLoadTracker::executing(LoadTracker::Demand &demand, int gpu_id) {
    delta_loadweights += demand.loadweights_size;
    demand.loadweights_size = 0;
    touched[gpu_id] = true;
}

void ModelLoadTracker::completed(LoadTracker::Demand &demand, int gpu_id) {
    delta_exec += demand.exec_size;
    demand.exec_size = 0;
    touched[gpu_id] = true;
}

void ModelLoadTracker::cancelled(LoadTracker::Demand &demand) {
    delta_loadweights += demand.loadweights_size;
    delta_exec += demand.exec_size;
    demand.loadweights_size = 0;
    demand.exec_size = 0;
}

void ModelLoadTracker::loadComplete(int gpu_id, bool success) {
    events.push({gpu_id, success});
}

/**
 * Attaches model priority from 'detached' to GPU by inserting 'priority' into GPU's 'cached' and 'not_cached'
 *
 * @param gpu The GPU which to attach to
 *
 * @throws None
 */
void LoadTracker::attach(GPU &gpu) {
    for (auto &priority : gpu.detached) {
        CHECK(priority->detached) << "Attaching model already attached";

        // Put back in to priority queues
        if (priority->model->loading[gpu.id]) {
            // Loading on a GPU is neither loadable nor evictable
            std::cout << "Attach: model with id " << priority->model->id << " is loading on gpu" << std::endl;
        } else if (priority->model->gpus[gpu.id]) {  // if model is loaded
            std::cout << "Attach: insert model with id " << priority->model->id << " in gpu cached" << std::endl;
            gpu.cached.insert(priority);
        } else {
            std::cout << "Attach: insert model with id " << priority->model->id << " in gpu not cached" << std::endl;
            gpu.not_cached.insert(priority);
        }

        priority->detached = false;
    }

    gpu.detached.clear();
}

/**
 * Detaches the model from the priority queues of all GPUs.
 *
 * @param model The model to detach from the priority queues
 *
 * @throws None
 */
void LoadTracker::detach(Model &model) {
    // Remove from priority queues
    for (unsigned i = 0; i < n_gpus; i++) {
        auto &gpu = gpus[i];
        auto &priority = model.priorities[i];

        // Only detach once
        if (priority->detached) continue;
        priority->detached = true;
        gpu.detached.push_back(priority);

        if (model.loading[i]) {
            // Loading on a GPU is neither loadable nor evictable
        } else if (model.gpus[i]) {
            auto it = gpu.cached.find(priority);
            CHECK(it != gpu.cached.end()) << "Thought we were cached when we weren't";

            gpu.cached.erase(it);
        } else {
            auto it = gpu.not_cached.find(priority);
            CHECK(it != gpu.not_cached.end()) << "Thought we were not cached when we were";

            gpu.not_cached.erase(it);
        }
    }

}

/**
 * Invalidates the priorities of a model if it's not stale, marking it as stale and adding it to the stale list for priority refreshing.
 *
 * @param model the model to invalidate priorities for
 *
 * @throws None
 */
void LoadTracker::invalidatePriorities(Model &model) {
    if (model.stale) return;
    model.stale = true;
    stale.push_back(&model);
}

/**
 * Refreshes the priorities of the models in the stale list.
 *
 * This function iterates over each model in the stale list and calls the
 * updatePriority function to update the priority of the model. It then sets
 * the stale flag of the model to false. After updating all the models, the
 * stale list is cleared.
 *
 * @throws None
 */
void LoadTracker::refreshPriorities() {
    for (auto &model : stale) {
        updatePriority(*model);
        model->stale = false;
    }
    stale.clear();
}

/**
 * Updates the priority of a model based on its current state and the state of its GPUs.
 *
 * @param model The model whose priority is to be updated.  
 *
 * @throws std::logic_error If the model is not stale.
 *
 * @post The priority of each priority in the model's priorities vector is updated based on the model's state and the state of its GPUs.
 */
void LoadTracker::updatePriority(Model &model) {
    CHECK(model.stale) << "Updating priority on non-stale model";

    // Calculate each GPU's weight
    double total_weight = 0;
    for (unsigned i = 0; i < n_gpus; i++) {
        if (model.gpus[i]) {  // if model is loaded on that gpu
            total_weight += gpus[i].weight;
        }
        CHECK(model.priorities[i]->detached) << "Updating priority on attached model";
    }

    // Load priority is calculated differently to evict priority
    // First, load priority.  Load priority is simply whether we can satisfy outstanding_loadweights
    int64_t load_priority = model.outstanding_loadweights;

    // Subtract served load
    if (total_weight > 0 && load_priority > 0) {
        for (unsigned i = 0; i < n_gpus; i++) {
            if (!model.gpus[i]) continue; // Skip models we are not loaded on

            int64_t required = model.outstanding_loadweights * (gpus[i].weight / total_weight);
            int64_t served = (capacity * required) / gpus[i].outstanding;
            load_priority -= served;
        }
    }

    bool is_empty = model.outstanding_loadweights == 0 && model.outstanding_exec == 0;

    for (unsigned i = 0; i < n_gpus; i++) {
        if (model.gpus[i]) {
            model.priorities[i]->priority = model.last_used[i];
        } else {
            model.priorities[i]->priority = load_priority;
        }
        model.priorities[i]->is_empty = is_empty;
        model.priorities[i]->last_used = model.last_used[i];
    }
}

void LoadTracker::clearLoad(Model &model) {
    for (unsigned i = 0; i < n_gpus; i++) {
        gpus[i].outstanding -= model.allocations[i];
        model.allocations[i] = 0;
    }
}

/**
 * Calculates the total_weight of all the GPUs and distributes the allocation of the model on each GPU.
 *
 * @param model Reference to the Model object on which the load distribution is performed
 */
void LoadTracker::distributeLoad(Model &model) {
    // Update all the counters
    model.outstanding_exec -= model.completed_exec;
    model.completed_exec = 0;
    int64_t loadweights_delta = std::max(model.completed_loadweights, model.timedout_loadweights);
    model.outstanding_loadweights -= loadweights_delta;
    model.completed_loadweights -= loadweights_delta;
    model.timedout_loadweights -= loadweights_delta;

    clearLoad(model);

    if (model.gpu_count == 0) return;

    // For demand tracking we use exec

    double total_weight = 0;
    for (unsigned i = 0; i < n_gpus; i++) {
        if (model.gpus[i]) {
            total_weight += gpus[i].weight;
        }
    }

    for (unsigned i = 0; i < n_gpus; i++) {
        if (model.gpus[i]) {
            auto allocation = model.outstanding_exec * (gpus[i].weight / total_weight);
            model.allocations[i] = allocation;
            gpus[i].outstanding += allocation;
            gpus[i].weight = capacity / ((double) gpus[i].outstanding);
        }
    }
}

/**
 * Update flags indicating that the model is on the GPU and that it is loading on the GPU
 *
 * @param model The model loaded on gpu
 * @param gpu The GPU loading the model
 *
 * @throws std::logic_error If the model is already on the GPU, if the model is already loading on the GPU, or if the GPU already thinks it has the model.
 */
void LoadTracker::addGPU(Model &model, GPU &gpu) {
    CHECK(!model.gpus[gpu.id]) << "Adding model to GPU that already has it";
    CHECK(!model.loading[gpu.id]) << "Adding model to GPU that is already loading it";
    CHECK(!gpu.models[model.id]) << "Adding model to GPU that thinks it already has it";

    // std::cout << "addGPU: model: " << model.id << ", gpu: " << gpu.id << std::endl;

    model.gpus[gpu.id] = true;
    model.loading[gpu.id] = true;
    gpu.models[model.id] = true;
    model.priorities[gpu.id]->preference = model.gpu_count++;
    model.last_used[gpu.id] = seqno_seed++;
}

void LoadTracker::addGPUcomplete(Model &model, GPU &gpu) {
    CHECK(model.gpus[gpu.id]) << "Model load completed on GPU that didn't expect it";
    CHECK(gpu.models[model.id]) << "Model load completed on GPU that didn't expect it";
    CHECK(model.loading[gpu.id]) << "Model load completed on GPU that wasn't loading";

    // std::cout << "addGPUcomplete: model: " << model.id << ", gpu: " << gpu.id << std::endl;

    model.loading[gpu.id] = false;
    model.last_used[gpu.id] = seqno_seed++;
}

/**
 * Removes a model from GPU and updates the model and GPU state accordingly.
 *
 * @param model The model to remove the GPU from.
 * @param gpu The GPU to remove.
 * @param evicted Whether the GPU was evicted or not.
 *
 * @throws None
 */
void LoadTracker::removeGPU(Model &model, GPU &gpu, bool evicted) {
    CHECK(model.gpus[gpu.id]) << "Removing Model from GPU that doesn't have it";
    CHECK(gpu.models[model.id]) << "Removing Model from GPU that doesn't think it has it";
    if (evicted) {
        CHECK(!model.loading[gpu.id]) << "Evicted loading model";
    } else {
        CHECK(model.loading[gpu.id]) << "Evicted model that is not loading";
    }
    
    model.gpus[gpu.id] = false;
    model.loading[gpu.id] = false;
    gpu.models[model.id] = false;
    model.gpu_count--;
    for (unsigned i = 0; i < n_gpus; i++) {
        auto pref = model.priorities[gpu.id]->preference;
        if (model.priorities[i]->preference > pref) {
            model.priorities[i]->preference--;
        }
        if (model.gpus[i]) {
            model.priorities[i]->last_used = seqno_seed++;
        }
    }
}

/**
 * Looks for timed out requests, then detaches its models from GPUs, invalidates model priorities,
 * and then re-distributes the load of the models on the GPUs accordingly.
 *
 * @param now the current time
 *
 * @throws None
 */
void LoadTracker::checkRequests(uint64_t now) {
    while (!requests.empty() && requests.top().time < now) {
        auto &request = requests.top();
        auto &model = models[request.model_id];
        model.timedout_loadweights += request.loadweights_size;

    	detach(model);
        invalidatePriorities(model);
        distributeLoad(model);

        requests.pop();
    }
}

LoadTracker::LoadTracker(int num_gpus, int num_models, uint64_t capacity) : 
n_models(num_models), n_gpus(num_gpus), capacity(capacity) {
    stale.reserve(num_models);
    gpus.resize(num_gpus);
    for (unsigned i = 0; i < num_gpus; i++) {
        gpus[i].id = i;
        gpus[i].models.resize(num_models, false);
    }

    models.resize(num_models);
    for (unsigned i = 0; i < num_models; i++) {
        auto &model = models[i];
        model.id = i;
        model.gpus.resize(num_gpus, false);  // resize from size '0' to size 'num_gpus'
        model.loading.resize(num_gpus, false);
        model.allocations.resize(num_gpus, 0);
        model.last_used.resize(num_gpus, 0);
        for (unsigned i = 0; i < num_gpus; i++) {
            model.last_used[i] = seqno_seed++;
        }

        model.priorities.resize(num_gpus);
        for (unsigned j = 0; j < num_gpus; j++) {
            auto priority = new ModelPriority(&model);
            priority->last_used = model.last_used[j];
            model.priorities[j] = priority;

            std::cout << "Model " << model.id << " priority " << priority->preference << " on GPU (not cached) " << j << std::endl;

            gpus[j].not_cached.insert(priority);
        }
    }            
}

/**
 * Loads a model onto a GPU.
 *
 * @param gpu_id the ID of the GPU to load the model onto
 * @param requires_eviction whether to evict the least recently used model if the GPU is out of memory
 *
 * @return the ID of the loaded model, or -1 if the GPU is out of memory or if the model is empty or if all demand is satisfied
 *
 * @throws None
 */
int LoadTracker::loadModel(int gpu_id, bool requires_eviction) {
    // Complete any pending requests
    checkRequests(util::now());  // Remove timed out requests and re-distributes load of models on GPUs

    auto &gpu = gpus[gpu_id];

    // Update and re-enqueue all models
    refreshPriorities(); // update load priority of models
    attach(gpu);  // attach model to priority queues of gpu

    if (gpu.not_cached.size() == 0) {
        // std::cout << "loadModel error: gpu.not_cached has no models to load" << std::endl;
        return -1;  // if no models to load
    }
    auto &priority = *gpu.not_cached.begin();
    if (priority->is_empty) {  // bool is_empty = model.outstanding_loadweights == 0 && model.outstanding_exec == 0;
        // std::cout << "loadModel error: model priority is empty, model: " << priority->model->id << ", model.outstanding_loadweights: " << priority->model->outstanding_loadweights << ", model.outstanding_exec: " << priority->model->outstanding_exec << std::endl;
        return -1;  // // if no models to load
    }
    if (priority <= 0) {
        std::cout << "loadModel error: all demand satisfied" << std::endl;
        return -1; // all demand satisfied
    }
    // Load model
    Model &model = *(priority->model);
    // std::cout << "loadModel: detach" << std::endl;
    detach(model);  // remove model from priority queues
    // std::cout << "loadModel: invalidatePriorities" << std::endl;
    invalidatePriorities(model);  // mark model as 'stale'
    // std::cout << "loadModel: addGPU" << std::endl;
    addGPU(model, gpu);  // update flags to indicate loading
    // std::cout << "loadModel: distributeLoad" << std::endl;
    distributeLoad(model);  // distribute allocation of load on GPUs

    return model.id;
}

/**
 * Evicts a model from a GPU based on eviction policy (LRU)
 *
 * @param gpu_id the ID of the GPU from which to evict the model
 *
 * @return the ID of the evicted model, or -1 if no models are cached on the GPU
 *
 * @throws None
 */
int LoadTracker::evictModel(int gpu_id) {
    // Update and re-enqueue all models
    refreshPriorities();
    attach(gpus[gpu_id]);

    auto &gpu = gpus[gpu_id];
    if (gpu.cached.size() == 0) {
        return -1;
    }
    auto &priority = *gpu.cached.rbegin();
    Model &model = *(priority->model);

    detach(model);
    invalidatePriorities(model);
    removeGPU(model, gpus[gpu_id], true);
    distributeLoad(model);

    return model.id;
}

/**
 * Processes a ModelLoadTracker by detaching the tracker model from GPUs, invalidating model priorities (by adding it to 'stale' queue),
 * removing pending load demand (timed out requests), adding new requests (from tracker), processing completed requests, updating last
 * used gpus for models, processing loadcomplete events, and re-distributing the model's load on gpus accordingly.
 *
 * @param tracker pointer to the ModelLoadTracker to process
 *
 * @throws None
 */
void LoadTracker::process(ModelLoadTracker* tracker) {
    // Detach the model from gpus, aka remove it from cache, etc.
    Model& model = models[tracker->model_id];
    detach(model);
    invalidatePriorities(model); // add the model to the 'stale' queue

    // Remove pending load demand
    uint64_t now = util::now();
    checkRequests(now);  // remove timed out requests

    // Add new requests (from tracker)
    int64_t loadweights = tracker->new_loadweights.exchange(0);
    int64_t exec = tracker->new_exec.exchange(0);
    if (loadweights > 0) {
        LoadTracker::Request request;
        request.model_id = tracker->model_id;
        request.loadweights_size = tracker->new_loadweights;
        request.time = now + tracker->start_loadweights_by;
        requests.push(request);
    }
    model.outstanding_exec += exec;
    model.outstanding_loadweights += loadweights;

    // Process completed requests
    model.completed_exec += tracker->delta_exec.exchange(0);
    model.completed_loadweights += tracker->delta_loadweights.exchange(0);

    // Update last used for models
    for (int i = 0; i < tracker->touched.size(); i++) { // touched[i] == true: gpu i was used recently
        if (tracker->touched[i]) {
            model.last_used[i] = seqno_seed++;
            tracker->touched[i] = false;
        }
    }

    // Process loadcomplete events
    ModelLoadTracker::LoadEvent event;
    while (tracker->events.try_pop(event)) {
        if (event.success) {
            addGPUcomplete(model, gpus[event.gpu_id]); // update flags (model.loading, model.last_used)
        } else {
            removeGPU(model, gpus[event.gpu_id], false); // update flags (model.gpus, model.loading, gpu.models, model.gpu_count, model.priorities)
        }
    }
    
    // Re-distribute model's load
    distributeLoad(model);
}

}
}
}