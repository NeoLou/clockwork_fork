#include "clockwork/controller/infer5/infer5_scheduler.h"
#include "clockwork/action.h"
#include "cluster_comm.pb.h"
#include <grpcpp/impl/codegen/status.h>
#include <iostream>
#include <memory>
#include <ostream>
#include <unistd.h>
#include <vector>

namespace clockwork {
namespace scheduler {
namespace infer5 {

std::atomic_uint64_t action_id_seed = 0;


Scheduler::Scheduler(uint64_t default_slo, uint64_t latest_delta,
                     uint64_t schedule_ahead, 
                     bool generate_inputs, int max_gpus,
                     uint64_t max_allowable_exec_time, unsigned max_batch_size,
                     std::string actions_filename)
    : default_slo(default_slo),
      latest_delta(latest_delta),
      schedule_ahead(schedule_ahead),
      max_allowable_exec_time(max_allowable_exec_time),
      max_batch_size(max_batch_size),
      generate_inputs(generate_inputs),
      max_gpus(max_gpus),
      actions_filename(actions_filename),
      has_logged_inputs_status(ATOMIC_FLAG_INIT) {
    std::cout << "Infer5 Scheduler using:" << std::endl;
    std::cout << "\t default_slo=" << default_slo << std::endl;
    std::cout << "\t latest_delta=" << latest_delta << std::endl;
    std::cout << "\t schedule_ahead=" << schedule_ahead << std::endl;
    std::cout << "\t max_allowable_exec_time=" << max_allowable_exec_time << std::endl;
    std::cout << "\t max_batch_size=" << max_batch_size << std::endl;
    std::cout << "\t generate_inputs=" << generate_inputs << std::endl;
    std::cout << "\t max_gpus=" << max_gpus << std::endl;

    if (generate_inputs) {
        input_generator = new util::InputGenerator();
    }
}

Scheduler::RequestImpl::RequestImpl(
    Scheduler* scheduler,
    clientapi::InferenceRequest request,
    std::function<void(clientapi::InferenceResponse&)> callback) : 
        scheduler(scheduler),
        request(request), 
        callback(callback),
        locked(false),
        response_sent(ATOMIC_FLAG_INIT) {
    // Set the response header fields now
    response.header.user_request_id = request.header.user_request_id;
    response.header.message = "";
    response.header.status = 0;
    response.model_id = request.model_id;
    response.batch_size = request.batch_size;
    response.output = nullptr;
    response.output_size = 0;
}

Scheduler::RequestImpl::~RequestImpl() {
    delete static_cast<char*>(request.input);
}

void Scheduler::RequestImpl::lock() {
    locked = true;
}

void Scheduler::RequestImpl::set_model(Model* model) {
    this->model = model;
    response.arrival_count = model->copies_loaded;
}

void Scheduler::RequestImpl::set_slo(uint64_t default_slo) {
    slo = default_slo;  // default_slo = 100000000UL
    if (request.slo_factor > 0) {
        slo = model->b1_exec * request.slo_factor;
    }
    response.deadline = request.arrival + slo; // arrival = 0 by default

    exec_slo = std::min(slo, slo - Scheduler::buffer);  // buffer = 5000000UL by default, so exec_slo = 95000000UL
    exec_slo = std::max(exec_slo, scheduler->schedule_ahead + Scheduler::buffer); // scheduler_ahead = 10000000UL, sched + buffer = 15000000UL, so exec_slo = 95000000UL
    deadline = request.arrival + exec_slo;

    weights_slo = std::min(max_loadweights_slo, 
                           std::min(slo, 
                                      slo - (model->estimate_weights() + model->b1_exec + Scheduler::buffer + scheduler->schedule_ahead)));
    weights_slo = std::max(weights_slo, scheduler->schedule_ahead + Scheduler::buffer);
}

void Scheduler::RequestImpl::set_result(char* output, size_t output_size) {
    response.header.status = clockworkSuccess;
    response.output = output;
    response.output_size = output_size;
    response.departure_count = model->copies_loaded;
}

void Scheduler::RequestImpl::set_error(int status, std::string message) {
    response.header.status = status;
    response.header.message = message;
}

bool Scheduler::RequestImpl::complete(uint64_t now, int gpu_id) {
    if (print_debug) std::cout << ("Client <--  " + response.str() + "\n");

    // Here to ensure only one response is sent
    if (response_sent.test_and_set()) return false;

    // Set the departure time (controller.cpp can also do this, 
    // but we want to report departure time back to the action to determine goodput)
    response.departure = now;

    callback(response);

    return response.header.status == clockworkSuccess && response.departure <= response.deadline;
}

void Scheduler::RequestImpl::timeout() {
    if (locked) return;

    if (print_debug) std::cout << ("Client <--  " + response.str() + "\n");

    // Here to ensure only one response is sent
    if (response_sent.test_and_set()) return;

    if (response.header.status == 0)
        response.header.status = clockworkTimeout;
    response.departure = util::now();
    response.departure_count = model->copies_loaded;

    callback(response);

    model->tracker->cancelled(demand);
    model->invalidate_tracker();
}

void Scheduler::RequestImpl::finalize() {
    timeout();
}

unsigned Scheduler::Model::batch_lookup(unsigned num_requests) {
    return num_requests > max_batch_size ? max_batch_size : batch_lookup_[num_requests];
}

unsigned Scheduler::Model::index_lookup(unsigned batch_size) {
    return batch_size > max_batch_size ? max_batch_size : index_lookup_[batch_size];
}

Scheduler::Model::Model(Scheduler* scheduler, BatchedModelState &state)
    : scheduler(scheduler),
      id(state.id), 
      num_weights_pages(state.num_weights_pages),
      input_size(state.input_size),
      output_size(state.output_size),
      stale(ATOMIC_FLAG_INIT) {
    for (auto &batch_size : state.supported_batch_sizes) {

        uint64_t estimate = 100000UL; // Default 0.1ms estimate

        // Lookup real estimate if it was provided
        auto it = state.exec_duration.find(batch_size);
        if (it != state.exec_duration.end()) {
            estimate = it->second;
        }

        // Limit the batch sizes we use
        if (batch_size == 1 || 
            (estimate > 0 
                && estimate <= scheduler->max_allowable_exec_time
                && batch_size <= scheduler->max_batch_size)) {
            if (estimates.size() < batch_size + 1) {
                estimates.resize(batch_size+1, 100000UL * Scheduler::default_clock);
            }
            estimates[batch_size] = estimate * Scheduler::default_clock;
            estimators[batch_size] = new SlidingWindow(Scheduler::estimate_window_size);
            supported_batch_sizes.push_back(batch_size);
            queues.push_back(new ModelQueue(batch_size));
        } else {
            std::cout << "Excluding b" << batch_size << " with estimate " << estimate << "model=" << state.model_path << std::endl;
        }

    }

    weights_estimator = new SlidingWindow(Scheduler::estimate_window_size);
    weights_estimate = state.weights_transfer_duration;

    batch_lookup_ = util::make_batch_lookup(supported_batch_sizes);
    index_lookup_ = util::make_reverse_batch_lookup(supported_batch_sizes);
    max_batch_size = batch_lookup_.size() - 1;

    b1_exec = estimate(1); // Use this for slo_factor
}

void Scheduler::Model::enqueue(Request request) {
    request->id = request_id_seed_++;
    incoming_requests.push(request);
    requests_queued++;

    for (auto &instance : instances) {
        instance->activate();
    }
}

void Scheduler::Model::pull_incoming_requests() {
    Request request;
    while (incoming_requests.try_pop(request)) {
        request->seqno = seqno_seed++;
        for (auto &queue : queues) {
            queue->push(request);
        }
    }
}

/**
 * Pulls incoming requests and creates a strategy for each queue of request in 'queues'.
 * The strategy sets the priority of a queue as 'deadline of the first request'  - 'estimate of execution'.
 *
 * @param gpu_id The ID of the GPU.
 * @param gpu_clock The GPU clock.
 * @param max_batchsize The maximum batch size.
 * @return A vector of StrategyImpl objects representing the strategies.
 */
std::vector<Scheduler::StrategyImpl> Scheduler::Model::new_strategies(int gpu_id, unsigned gpu_clock, int max_batchsize) {
    tbb::queuing_mutex::scoped_lock lock(mutex);

    pull_incoming_requests();  // pull requests into 'queues'

    std::vector<StrategyImpl> strategies;
    for (int i = queues.size()-1; i >= 0; i--) {
        auto &queue = queues[i];

        if (queue->size() == 0) continue;
        if (queue->batchsize > max_batchsize) continue;  // drops requests that have too large batchsize

        StrategyImpl strategy;
        strategy.priority = queue->front()->deadline - estimate(queue->batchsize);
        strategy.batch_size = queue->batchsize;
        strategy.instance = instances[gpu_id];
        strategies.push_back(strategy);
    }

    return strategies;
}

/**
 * Attempt to dequeue requests from queues based on specified conditions.
 *
 * @param free_at the timestamp representing the current time
 * @param gpu_clock the current clock speed of the GPU
 * @param min_batchsize the minimum batch size requested
 *
 * @return a pointer to the dequeued InferAction object or nullptr if conditions are not met
 */
Scheduler::InferAction* Scheduler::Model::try_dequeue(
        uint64_t free_at,
        unsigned gpu_clock,
        int min_batchsize)
{   
    // Drain all requests from all queues that wouldn't be satisfiable
    // If specified batchsize is not achievable, return
    // Use maximum possible batch size
    // Drain requests from all queues to catch up to id of drained req
    // TODO: properly maintain requests_queued counter
    tbb::queuing_mutex::scoped_lock lock(mutex);

    // Drain incoming requests to batchsize queues (for now there are 4 batch queues for 1, 2, 4, 8)
    pull_incoming_requests();

    uint64_t size_before = queues[0]->size();

    // Drop requests from batchsize queues that won't complete in time
    for (auto &queue : queues) {
        uint64_t exec_time = estimate(queue->batchsize, gpu_clock);
        uint64_t completion_time = free_at + exec_time;

        while (queue->size() > 0 && queue->front()->deadline < completion_time) {
            queue->pop();
        }
    }
    // Find the appropriate queue
    int i = 0;
    while (i < queues.size()-1 && queues[i+1]->has_demand()) { // If size of queue is more or equal to batchsize, we have demand
        i++;
    }

    // Not enough requests available at the requested batchsize
    auto &queue = queues[i];
    //std::cout << "size before" << size_before << ", size of queue [" << i << "]: " << queue->size() << std::endl;
    if (queue->batchsize < min_batchsize) {
        //std::cout << "Not enough requests available, batchsize = " << queue->batchsize << ", min_batchsize = " << min_batchsize << std::endl;
        return nullptr;
    }
    if (!queue->has_demand()) { // If size of queue is more or equal to batchsize, has_demand = true
        // std::cout << "Queue has no demand" << std::endl;
        return nullptr;
    }

    // Create the action
    uint64_t seqno;
    auto action = new InferAction(scheduler, this);
    for (unsigned i = 0; i < queue->batchsize; i++) {
        auto &request = queue->front();
        request->lock();
        action->requests.push_back(request);
        seqno = request->seqno;
        queue->pop();
    }
    action->set_expectations(free_at, estimate(queue->batchsize, gpu_clock), gpu_clock);
    action->batch();

    // Catch up the queues
    for (auto &queue : queues) {
        while (queue->size() > 0 && queue->front()->seqno <= seqno) {
            queue->pop();
        }
    }

    uint64_t size_after = queues[0]->size();
    requests_queued -= (size_before - size_after);
    // std::cout << "size_after: " << size_after << ", requests_queued: " << requests_queued << std::endl;

    return action;
}

const float Scheduler::estimate_percentile = 0.99;

void Scheduler::Model::add_measurement(unsigned batch_size, uint64_t duration, unsigned gpu_clock) {
    tbb::queuing_mutex::scoped_lock lock(estimates_mutex);

    auto it = estimators.find(batch_size);
    CHECK(it != estimators.end()) << "Unsupported batch size " << batch_size;
    auto estimator = it->second;
    estimator->insert(duration * gpu_clock);

    estimates[batch_size] = estimator->get_percentile(Scheduler::estimate_percentile);
}

void Scheduler::Model::add_weights_measurement(uint64_t duration) {
    tbb::queuing_mutex::scoped_lock lock(weights_estimate_mutex);

    weights_estimator->insert(duration);
    weights_estimate = weights_estimator->get_percentile(Scheduler::estimate_percentile);
}

uint64_t Scheduler::Model::estimate(unsigned batch_size) {
    return Scheduler::Model::estimate(batch_size, Scheduler::default_clock);
}

/**
* Estimates the duration of a model execution for a given batch size and clock speed.
*
* @param batch_size the size of the input batch
* @param clock the clock speed at which the execution will be performed
*
* @return the estimated duration of the model execution in nanoseconds
*
* @throws Non
*/
uint64_t Scheduler::Model::estimate(unsigned batch_size, int clock) {
    unsigned effective_batch_size = batch_lookup(batch_size);
    return estimates[effective_batch_size] / clock;
}

uint64_t Scheduler::Model::estimate_weights() {
    return weights_estimate;
}

Scheduler::InferAction::InferAction(Scheduler* scheduler, Model* model) : scheduler(scheduler), model(model) {
    action->id = action_id_seed++;
    action->model_id = model->id;
}

Scheduler::InferAction::~InferAction() {
    if (result != nullptr) {
        delete result->output;
        delete action->input;
    }
}


void Scheduler::InferAction::batch() {
    action->batch_size = requests.size();
    action->input_size = 0;

    if (!scheduler->has_logged_inputs_status.test_and_set()) {
        std::stringstream msg;
        if (requests[0]->request.input_size == 0) {
            if (scheduler->generate_inputs) {
                msg << "Network Status:  Client ✘✘✘✘✘ Controller ✔✔✔✔✔ Workers (inputs generated by controller)";
            } else {
                msg << "Network Status:  Client ✘✘✘✘✘ Controller ✘✘✘✘✘ Workers (inputs generated by worker)";                
            }
        } else {
            msg << "Network Status:  Client ✔✔✔✔✔ Controller ✔✔✔✔✔ Workers (inputs generated by client)";   
        }
        msg << std::endl;
        // std::cout << msg.str();
    }

    for (auto &req : requests) {
        auto &r = req->request;
        if (r.input_size == 0 && scheduler->generate_inputs) {
            generated_inputs = true;
            char* generated_input;
            scheduler->input_generator->generatePrecompressedInput(model->input_size, &generated_input, &r.input_size);
            r.input = generated_input;
        }
        action->input_size += r.input_size;
    }

    action->input = new char[action->input_size];
    size_t offset = 0;
    for (auto &req : requests) {
        auto &r = req->request;
        std::memcpy(action->input + offset, r.input, r.input_size);
        offset += r.input_size;
        action->input_sizes.push_back(r.input_size);
    }
}

void Scheduler::InferAction::unbatch() {
    size_t single_output_size = result->output_size / requests.size();
    if (generated_inputs) single_output_size = 0;
    size_t offset = 0;
    for (unsigned i = 0; i < requests.size(); i++) {
        char* output = new char[single_output_size];
        std::memcpy(output, result->output + offset, single_output_size);
        offset += single_output_size;

        requests[i]->set_result(output, single_output_size);
    }
}

void Scheduler::InferAction::set_error(std::shared_ptr<workerapi::ErrorResult> &error) {
    this->error = error;
    for (auto &request : requests) {
        request->set_error(error->status, error->message);
    }
}

void Scheduler::InferAction::set_result(std::shared_ptr<workerapi::InferResult> &result) {
    this->result = result;
    this->unbatch();
}

float Scheduler::InferAction::complete(uint64_t now, int gpu_id) {
    float successful_requests = 0;
    float total_requests = 0;
    for (auto &request : requests) {
        if (request->complete(now, gpu_id)) {
            successful_requests += 1;
        }
        model->tracker->completed(request->demand, gpu_id);
        model->invalidate_tracker();

        total_requests += 1;
    }

    return successful_requests / total_requests;
}

void Scheduler::InferAction::set_expectations(uint64_t exec_start, uint64_t duration, int clock) {
    uint64_t now = util::now();
    action->expected_duration = duration;
    action->expected_exec_complete = exec_start + duration;
    action->expected_gpu_clock = clock;
    // action->earliest = exec_start;
    // action->latest = action->earliest + Scheduler::latest_delta;
    // action->earliest = std::max(util::now() + future, exec_start - Scheduler::latest_delta);
    action->earliest = now;
    action->latest = std::max(now + future + scheduler->latest_delta, exec_start + scheduler->latest_delta);
    send_by = std::max(now + future, action->latest - 5000000UL);
    report_error_at = action->latest + duration;
    // action->earliest = util::now() - Scheduler::schedule_ahead;
    // action->latest = action->expected_exec_complete + Scheduler::latest_delta;
}

        // ModelInstance* instance;
        // unsigned version;
        // ControllerActionTelemetry telemetry;
        // std::shared_ptr<workerapi::LoadWeights> action = std::make_shared<workerapi::LoadWeights>();
        // std::shared_ptr<workerapi::ErrorResult> error = nullptr;
        // std::shared_ptr<workerapi::LoadWeightsResult> result = nullptr;

Scheduler::LoadWeightsAction::LoadWeightsAction(
    Scheduler* scheduler, ModelInstance* instance)
      : scheduler(scheduler), instance(instance) {
    action->id = action_id_seed++;
    action->model_id = instance->model->id;
}

void Scheduler::LoadWeightsAction::set_expectations(uint64_t exec_start, uint64_t duration) {
    uint64_t now = util::now();
    action->expected_duration = duration;
    action->expected_exec_complete = exec_start + duration;
    // action->earliest = std::max(now + future, exec_start - scheduler->latest_delta);
    action->earliest = now;
    action->latest = std::max(now + future + scheduler->latest_delta, exec_start + scheduler->latest_delta);
}

void Scheduler::LoadWeightsAction::set_error(std::shared_ptr<workerapi::ErrorResult> &error) {
    this->error = error;
    if (version = instance->version) {
        instance->version++;
        instance->loading = false;
        instance->loaded = false;
    }
}

void Scheduler::LoadWeightsAction::set_result(std::shared_ptr<workerapi::LoadWeightsResult> &result) {
    this->result = result;
    if (version == instance->version) {
        std::cout << "Loaded model " << instance->model->id << " version " << instance->version << std::endl;
        instance->loaded = true;
        instance->loading = false;
    }
}

Scheduler::EvictWeightsAction::EvictWeightsAction(ModelInstance* instance) : instance(instance) {
    action->id = action_id_seed++;
    action->model_id = instance->model->id;
}

void Scheduler::EvictWeightsAction::set_expectations() {
    action->earliest = 0;
    action->latest = UINT64_MAX;
}

void Scheduler::EvictWeightsAction::set_error(std::shared_ptr<workerapi::ErrorResult> &error) {
    this->error = error;
    CHECK(false) << "Error in EvictWeightsAction" << error->str();
}

void Scheduler::EvictWeightsAction::set_result(std::shared_ptr<workerapi::EvictWeightsResult> &result) {
    this->result = result;
}

void Scheduler::ModelInstance::activate() {
    if (!active.test_and_set()) {
        gpu->activated.push(this);
    }
}

void Scheduler::ModelInstance::deactivate() {
    active.clear();
}

Scheduler::GPU::GPU(
    unsigned id,
    Scheduler* scheduler, 
    network::controller::WorkerConnection* worker,
    unsigned worker_id,
    unsigned gpu_id,
    unsigned pages) 
    : id(id),
      scheduler(scheduler),
      worker(worker),
      worker_id(worker_id),
      gpu_id(gpu_id),
      pages(pages),
      free_pages(pages),
      exec(Scheduler::default_clock, Scheduler::lag, Scheduler::future), 
      loadweights(Scheduler::default_clock, Scheduler::lag, Scheduler::future) {
}

void Scheduler::GPU::grpc_send_action(InferAction* action, uint64_t send_by) {
    uint64_t now = util::now();
    if (send_by < now) {
        std::cout << "Could not send action to worker in time, action->send_by: " << send_by << " now: " << now << "" << std::endl;
        return;
    }
    cluster_comm::BatchedInferRequest batched_infer_req;
    batched_infer_req.set_batch_id(action->action->id);
    std::vector<cluster_comm::InferRequest> infer_reqs;
    for (std::shared_ptr<RequestImpl> req : action->requests) {
        // Create tensor proto object
        cluster_comm::Tensor input_tensor;
        input_tensor.set_data(req->request.input, req->request.input_size);
        *input_tensor.mutable_shape() = {req->request.input_shape.begin(), req->request.input_shape.end()};

        // Populate proto request
        cluster_comm::InferRequest infer_req;
        infer_req.set_request_id(req->request.uuid);
        infer_req.set_request_local_id(req->id);
        infer_req.set_client_id(req->request.header.user_id);
        infer_req.set_model_id(req->model->id);
        infer_req.set_slo(req->slo);
        *infer_req.mutable_input_tensor() = input_tensor;

        infer_reqs.push_back(infer_req);
    }
    *batched_infer_req.mutable_requests() = {infer_reqs.begin(), infer_reqs.end()};
    
    // Code for async send request (skip error check for now, implement in future)
    // Note: I'm not sure this is 100% async, need to check
    grpc::ClientContext context;
    grpc::CompletionQueue cq;
    cluster_comm::Empty reply;
    grpc::Status status;

    uint64_t time_message_sent = util::now();
    batched_infer_req.set_time_message_sent(time_message_sent);

    std::unique_ptr<grpc::ClientAsyncResponseReader<cluster_comm::Empty>> rpc(
        stub_->AsyncsendBatchInferReq(&context, batched_infer_req, &cq));
    rpc->Finish(&reply, &status, (void*)1);

    // Error check
    // void* got_tag;
    // bool ok = false;
    // CHECK(cq.Next(&got_tag, &ok));
    // CHECK_EQ(got_tag, (void*)1);
    // CHECK(ok);
    // if (status.ok()) {
    //   std::cout << "RPC successful" << std::endl;
    // } else {
    //   std::cout << "RPC failed" << std::endl;
    //   std::cout << status.error_code() << ": " << status.error_message()
    //             << std::endl;
    // }
}

void Scheduler::GPU::grpc_send_action(LoadWeightsAction* action, uint64_t send_by) {
    uint64_t now = util::now();
    if (send_by < now) {
        std::cout << "Could not send action to worker in time, action->send_by: " << send_by << " now: " << now << "" << std::endl;
        return;
    }
    /* Data types
    message ModelPlacement {
        int32 model_id = 1;
        string model_name = 2;
        int32 local_gpu_id = 3;
        float resource_share = 4;
        int32 batch_size = 5;
        int32 slo = 6;
        int32 assigned_mem = 7;
        int32 assigned_sm = 8;
    }
    */
    cluster_comm::ModelPlacement model_placement;
    model_placement.set_model_id(action->action->model_id);
    model_placement.set_local_gpu_id(action->action->gpu_id);
    model_placement.set_resource_share(100.0f); // hard code for now
    model_placement.set_batch_size(1); // hard code for now
    model_placement.set_action_id(action->action->id);
    
    // Code for async send request (skip error check for now, implement in future)
    // Note: I'm not sure this is 100% async, need to check
    grpc::ClientContext context;
    grpc::CompletionQueue cq;
    cluster_comm::Empty reply;
    grpc::Status status;
    std::unique_ptr<grpc::ClientAsyncResponseReader<cluster_comm::Empty>> rpc(
        stub_->AsyncloadModel(&context, model_placement, &cq));
    rpc->Finish(&reply, &status, (void*)1);

    // Error check
    // void* got_tag;
    // bool ok = false;
    // CHECK(cq.Next(&got_tag, &ok));
    // CHECK_EQ(got_tag, (void*)1);
    // CHECK(ok);
    // if (status.ok()) {
    //   std::cout << "RPC successful" << std::endl;
    // } else {
    //   std::cout << "RPC failed" << std::endl;
    //   std::cout << status.error_code() << ": " << status.error_message()
    //             << std::endl;
    // }
}

void Scheduler::GPU::grpc_send_action(EvictWeightsAction* action, uint64_t send_by) {
    uint64_t now = util::now();
    if (send_by < now) {
        std::cout << "Could not send action to worker in time, action->send_by: " << send_by << " now: " << now << "" << std::endl;
        return;
    }
    
    /* Data types
    message ModelPlacement {
        int32 model_id = 1;
        string model_name = 2;
        int32 local_gpu_id = 3;
        float resource_share = 4;
        int32 batch_size = 5;
        int32 slo = 6;
        int32 assigned_mem = 7;
        int32 assigned_sm = 8;
    }
    */

    cluster_comm::ModelPlacement model_placement;
    model_placement.set_model_id(action->action->model_id);
    model_placement.set_local_gpu_id(action->action->gpu_id);
    model_placement.set_resource_share(100.0f); // hard code for now
    model_placement.set_batch_size(1); // hard code for now
    model_placement.set_action_id(action->action->id);

    // Code for async send request (skip error check for now, implement in future)
    // Note: I'm not sure this is 100% async, need to check
    grpc::ClientContext context;
    grpc::CompletionQueue cq;
    cluster_comm::Empty reply;
    grpc::Status status;
    std::unique_ptr<grpc::ClientAsyncResponseReader<cluster_comm::Empty>> rpc(
        stub_->AsyncevictModel(&context, model_placement, &cq));
    rpc->Finish(&reply, &status, (void*)1);

    // Error check
    // void* got_tag;
    // bool ok = false;
    // CHECK(cq.Next(&got_tag, &ok));
    // CHECK_EQ(got_tag, (void*)1);
    // CHECK(ok);
    // if (status.ok()) {
    //   std::cout << "RPC successful" << std::endl;
    // } else {
    //   std::cout << "RPC failed" << std::endl;
    //   std::cout << status.error_code() << ": " << status.error_message()
    //             << std::endl;
    // }
}

void Scheduler::GPU::send_action(InferAction* action) {
    auto &infer = action->action;
    infer->gpu_id = gpu_id;
    infer->worker_id = worker_id;

    // Update GPU state
    {
        tbb::queuing_mutex::scoped_lock lock(exec_mutex);
        exec.add(infer->id, infer->expected_duration);
    }

    // Save the callback
    auto callback = [this, action](std::shared_ptr<workerapi::Result> result) {
        this->infer_result(action, result);
    };
    scheduler->add_callback(infer->id, callback);

    // Record the telemetry
    action->telemetry.set(infer);
    action->telemetry.requests_queued = action->model->requests_queued;
    action->telemetry.copies_loaded = action->model->copies_loaded;

    // Immediately mark the requests as executing for load balancer
    for (auto &request : action->requests) {
        action->model->tracker->executing(request->demand, id);
    }
    action->model->invalidate_tracker();

    // Send the action
    // use our own C++ gRPC function to send action instead of clockwork network function
    this->grpc_send_action(action, action->send_by);

    // if action not sent by time "send_by", don't send and report error at time "report_error_at"
    // scheduler->network->send(worker, infer, action->send_by, action->report_error_at);

    if (print_debug) std::cout << ("Worker <--  " + infer->str() + "\n");
}

void Scheduler::GPU::send_action(LoadWeightsAction* action) {
    auto &load = action->action;
    load->gpu_id = gpu_id;
    load->worker_id = worker_id;

    // Update PCI state
    {
        tbb::queuing_mutex::scoped_lock lock(loadweights_mutex);
        loadweights.add(load->id, load->expected_duration);
    }

    // Save the callback
    auto callback = [this, action](std::shared_ptr<workerapi::Result> result) {
        this->load_result(action, result);
    };
    scheduler->add_callback(load->id, callback);

    // Record the telemetry
    action->telemetry.set(load);
    action->telemetry.requests_queued = action->instance->model->requests_queued;
    action->telemetry.copies_loaded = action->instance->model->copies_loaded;

    // Send the action
    // use our own C++ gRPC function to send action instead of clockwork network function
    this->grpc_send_action(action, UINT64_MAX);

    // scheduler->network->send(worker, load, UINT64_MAX, 0);


    if (print_debug || print_loads) std::cout << ("Worker <--  " + load->str() + "\n");
}

void Scheduler::GPU::send_action(EvictWeightsAction* action) {
    auto &evict = action->action;
    evict->gpu_id = gpu_id;
    evict->worker_id = worker_id;

    // Save the callback
    auto callback = [this, action](std::shared_ptr<workerapi::Result> result) {
        this->evict_result(action, result);
    };
    scheduler->add_callback(evict->id, callback);

    // Record the telemetry
    action->telemetry.set(evict);
    action->telemetry.requests_queued = action->instance->model->requests_queued;
    action->telemetry.copies_loaded = action->instance->model->copies_loaded+1;

    // Send the action
    // use our own C++ gRPC function to send action instead of clockwork network function
    this->grpc_send_action(action,UINT64_MAX);

    // scheduler->network->send(worker, evict, UINT64_MAX, 0);

    if (print_debug || print_loads) std::cout << ("Worker <--  " + evict->str() + "\n");    
}

/**
 * Evicts pages from the GPU to make space for the required number of pages.
 *
 * @param required_pages The number of pages required.
 *
 * @return A vector of EvictWeightsAction pointers, representing the evicted weights actions.
 *
 * @throws None
 */
std::vector<Scheduler::EvictWeightsAction*> Scheduler::GPU::evict_pages(unsigned required_pages) {
    std::vector<EvictWeightsAction*> ret;
    while (free_pages < required_pages) {  // keep evicting until we have enough pages
        int model_id = scheduler->tracker->evictModel(id);  // model to evict, based on LRU

        if (model_id == -1) break;

        ModelInstance* instance = instances[model_id];
        instance->version++;
        instance->loading = false;
        instance->loaded = false;
        instance->model->copies_loaded--;

        EvictWeightsAction* evict = new EvictWeightsAction(instance);
        evict->set_expectations();
        ret.push_back(evict);

        std::cout << "Free pages before eviction: " << free_pages << "\n";
        free_pages += scheduler->models[model_id]->num_weights_pages;
        std::cout << "Free pages after eviction: " << free_pages << "\n";
        eviction_required = true; // GPU reached capacity; evictions required in future
    }
    return ret;
}

/**
 * Schedules the loading of a model on the GPU.
 *
 * @return true if the model was successfully loaded, false otherwise.
 *
 * @throws None.
 */
bool Scheduler::GPU::schedule_load() {
    tbb::queuing_mutex::scoped_lock lock(load_mutex);

    uint64_t available;
    {
        tbb::queuing_mutex::scoped_lock lock(loadweights_mutex);  // 'loadweights' is a worker tracker
        available = loadweights.available();  // returns the time outstanding work will complete (aka time gpu will be available)
    }

    uint64_t now = util::now();
    if (available >= now + scheduler->schedule_ahead) {
        // std::cout << "Not scheduling load yet, available: " << available << " now: " << now << "\n";
        return false;  // schedule_ahead = 10ms by default
    }                                                             // don't schedule if gpu isn't available 10ms from now
    ModelInstance* instance;
    unsigned size;
    std::vector<EvictWeightsAction*> evict_actions;
    {
        tbb::queuing_mutex::scoped_lock load_lock(scheduler->tracker->load_mutex);
        tbb::queuing_mutex::scoped_lock lock(scheduler->tracker->mutex);

        int model_id = scheduler->tracker->loadModel(id, eviction_required);
        if (model_id == -1) {
            return false;
        }

        instance = instances[model_id];
        // std::cout << "creating load/evict action for model with id " << model_id << ", loaded: " << instance->loaded << ", loading: " << instance->loading << std::endl;

        CHECK(instance->loaded == false && instance->loading == false) << "Tracker asked to load model that is already loaded";

        size = scheduler->models[model_id]->num_weights_pages;
        evict_actions = evict_pages(size);  // prepare action to evict pages if necessary until we have enough pages
    }

    if (free_pages < size) {  // if size of model to load is too big for gpu
        instance->model->tracker->loadComplete(id, false);
        instance->model->invalidate_tracker();
    }

    // Send the evict actions (if any)
    for (auto &evict : evict_actions) {
        std::cout << "send evict action for model " << evict->action->model_id << std::endl;
        send_action(evict);
    }

    if (free_pages < size) { // if size of model to load is too big for gpu
        return false;
    }

    std::cout << "Free pages before loading: " << free_pages << std::endl;
    free_pages -= size;
    std::cout << "Free pages after loading: " << free_pages << std::endl;

    uint64_t expected_duration = instance->model->estimate_weights();

    LoadWeightsAction* action = new LoadWeightsAction(scheduler, instance);
    action->version = ++instance->version;
    instance->loading = true;
    instance->loaded = false;
    action->set_expectations(available, expected_duration);

    send_action(action);
    return true;
}

/**
 * Adds strategies for a single ModelInstance.
 * (stragety = setting priority of which job to do first)
 *
 * @param instance The ModelInstance for which to add strategies.
 * @param max_batchsize The maximum batch size.
 *
 * @return void
 *
 * @throws None
 */
void Scheduler::GPU::add_model_strategies(ModelInstance* instance, int max_batchsize) {
    instance->strategies = instance->model->new_strategies(id, exec.clock(), max_batchsize);
    if (instance->strategies.size() == 0) {
        // Strategy should be deactivated
        // There is an unimportant race condition here if requests are concurrently enqueued
        instance->deactivate(); // set 'active' flag to false
        return;
    }

    for (auto &strategy : instance->strategies) {
        this->strategies.insert(strategy);
    }
}

/**
* Schedules inference for a single GPU, by adding strategies for all activated model instances
*
* @return true if inference was scheduled, false otherwise
*
* @throws None
*/
bool Scheduler::GPU::schedule_infer() {
    // TODO: skip and deactivate models that have been evicted
    // TODO: what to do when we run out of reqs
    schedule_infer_count++;
    tbb::queuing_mutex::scoped_lock lock(infer_mutex);

    // All activated instances start dirty
    std::vector<ModelInstance*> newly_activated;
    ModelInstance* activated_model;
    // Pops all activated model instances into newly_activated
    while (activated.try_pop(activated_model)) {
        newly_activated.push_back(activated_model);
    }
    // Get and incorporate strategies for each activated instance
    for (auto &instance : newly_activated) {
        add_model_strategies(instance);
    }

    if (strategies.size() == 0) {
        schedule_infer_strategies_empty_count++;
    }

    // If model is empty, set active to false, then drain queue just in case
    bool active = false;
    while (strategies.size() > 0) {
        //std::cout << strategies.size() << " strategies scheduled for: " << newly_activated.size() << " newly activated models" << "\n";
        uint64_t exec_at;
        int clock;
        {
            tbb::queuing_mutex::scoped_lock lock(exec_mutex);
            exec_at = exec.available();  // exec_at = time outstanding work will complete (aka time that gpu will be available)
            clock = exec.clock();  // clock of worker
        }

        // schedule_until = how far we want to schedule until (in the future)
        uint64_t schedule_until = util::now() + scheduler->schedule_ahead; // schedule_ahead is 10ms by default
        if (exec_at >= schedule_until) {  // If time to schedule is to far in the future, don't schedule yet, keep looping
            // std::cout << "Not scheduling yet, exec_at: " << exec_at << ", schedule_until: " << schedule_until << "\n";
            schedule_infer_exec_full++;
            break;
        }

        auto strategy = *strategies.begin();  // Get the first strategy, strategies are a set sorted by priority,
        auto instance = strategy.instance; // then batch size, then model id

        // Remove all strategies for this instance (from instance and from gpu)
        int expected = instance->strategies.size();  // instance strategies
        int before = strategies.size();  // gpu strategies
        for (auto &strategy : instance->strategies) {
            auto it = strategies.find(strategy);  // find the gpu strategy that corresponds to the instance strategy
            CHECK(it != strategies.end()) << "Strategy missing from priority queue";
            strategies.erase(it);  // remove that strategy from gpu strategies
        }
        instance->strategies.clear();  // remove all instance strategies
        int after = strategies.size();

        // Deactivate evicted model
        if (!strategy.instance->loaded) {  // If model instance is not loaded, that means it has been evicted
            // std::cout << "Model not loaded: " << strategy.instance->model->id << "\n";
            strategy.instance->deactivate();
            continue;
        }

        // Attempt to dequeue requests from queues that will finish in time and can be batched
        InferAction* action = strategy.instance->model->try_dequeue(exec_at, clock, strategy.batch_size);

        schedule_infer_action_attempted++;
        if (action != nullptr) {
            schedule_infer_action_created++;
            send_action(action);
            active = true;
            add_model_strategies(strategy.instance);
            break;
        } else {
            //std::cout << "Action not created, instead create new strategies with smaller batch_size: " << strategy.batch_size-1 << "\n";
            add_model_strategies(strategy.instance, strategy.batch_size-1);
        }
    }

    if (active) {
        schedule_infer_active_count++;
    } else {
        schedule_infer_inactive_count++;
    }
    return active;
}

void Scheduler::GPU::infer_error(InferAction* action, std::shared_ptr<workerapi::ErrorResult> &error) {
    action->telemetry.set(error);
    
    // Update GPU state tracking
    {
        tbb::queuing_mutex::scoped_lock lock(exec_mutex);
        exec.error(error->id, util::now());
    }

    action->set_error(error);
    CHECK(action->complete(util::now(), id) == 0) << "ErrorResult should not result in successful requests";

    action->telemetry.goodput = 0;
    scheduler->printer->log(action->telemetry);

    delete action;
}

void Scheduler::GPU::infer_success(InferAction* action, std::shared_ptr<workerapi::InferResult> &result) {
    action->telemetry.set(result);

    // Update GPU state tracking
    {
        tbb::queuing_mutex::scoped_lock lock(exec_mutex);
        exec.success(result->id, result->exec.end);
        exec.update_clock(result->gpu_clock);
    }

    // Update model execution tracking
    action->model->add_measurement(
        action->action->batch_size, 
        result->exec.duration, 
        (result->gpu_clock + result->gpu_clock_before) / 2
    );

    action->set_result(result);
    action->telemetry.goodput = action->complete(util::now(), id);
    // scheduler->printer->log(action->telemetry);  // commented this out because we don't need it for orion

    delete action;
}

void Scheduler::GPU::infer_result(InferAction* action, std::shared_ptr<workerapi::Result> &result) {
    
    if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
        std::cout << "Calling infer_error\n";
        infer_error(action, error);

    } else if (auto infer = std::dynamic_pointer_cast<workerapi::InferResult>(result)) {
        infer_success(action, infer);

    } else {
        std::cout << "Unexpected result\n";
        CHECK(false) << "Unexpected response to Infer action" << result->str();

    }
}

void Scheduler::GPU::load_error(LoadWeightsAction* action, std::shared_ptr<workerapi::ErrorResult> &error){
    action->telemetry.set(error);
    action->telemetry.goodput = 0;
    action->set_error(error);

    // Track model status
    action->instance->model->tracker->loadComplete(id, false);
    action->instance->model->invalidate_tracker();
    free_pages += action->instance->model->num_weights_pages;

    // Update PCI state tracking
    {
        tbb::queuing_mutex::scoped_lock lock(loadweights_mutex);
        loadweights.error(error->id, util::now());
    }

    scheduler->printer->log(action->telemetry);

    delete action;
}

void Scheduler::GPU::load_success(LoadWeightsAction* action, std::shared_ptr<workerapi::LoadWeightsResult> &result) {
    action->telemetry.set(result);
    action->set_result(result);

    // Track model status
    action->instance->model->tracker->loadComplete(id, true);
    action->instance->model->invalidate_tracker();

    // Update PCI state tracking
    {
        tbb::queuing_mutex::scoped_lock lock(loadweights_mutex);
        loadweights.success(result->id, result->end);
    }

    // Update PCI tracking
    action->instance->model->add_weights_measurement(result->duration);
    action->instance->model->copies_loaded++;

    // Enable new requests
    action->instance->activate();

    // TODO: change this?
    action->telemetry.goodput = 1.0;

    // scheduler->printer->log(action->telemetry); // commented this out because we don't need it for orion

    delete action;
}

void Scheduler::GPU::load_result(LoadWeightsAction* action, std::shared_ptr<workerapi::Result> &result) {
    if (!print_debug && print_loads) std::cout << ("Worker  --> " + result->str() + "\n");

    if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
        load_error(action, error);

    } else if (auto load = std::dynamic_pointer_cast<workerapi::LoadWeightsResult>(result)) {
        load_success(action, load);

    } else {
        CHECK(false) << "Unexpected response to LoadWeights action" << result->str();

    }    
}

void Scheduler::GPU::evict_error(EvictWeightsAction* action, std::shared_ptr<workerapi::ErrorResult> &error){
    action->telemetry.set(error);

    action->set_error(error);

    scheduler->printer->log(action->telemetry);

    delete action;
}

void Scheduler::GPU::evict_success(EvictWeightsAction* action, std::shared_ptr<workerapi::EvictWeightsResult> &result) {
    action->telemetry.set(result);

    action->set_result(result);

    // scheduler->printer->log(action->telemetry); // commented this out because we don't need it for orion

    delete action;
}

void Scheduler::GPU::evict_result(EvictWeightsAction* action, std::shared_ptr<workerapi::Result> &result) {
    if (!print_debug && print_loads) std::cout << ("Worker  --> " + result->str() + "\n");

    if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
        evict_error(action, error);

    } else if (auto evict = std::dynamic_pointer_cast<workerapi::EvictWeightsResult>(result)) {
        evict_success(action, evict);

    } else {
        CHECK(false) << "Unexpected response to LoadWeights action" << result->str();

    }
}

void Scheduler::validate_clockwork_state(ClockworkState &state) {
    unsigned cache_size = state.workers[0].gpus[0].weights_cache_total_pages;
    std::cout << "----------------Checking if all GPUs have same cache_size: " << cache_size << std::endl;
    for (auto &worker : state.workers) {
        for (auto &gpu : worker.gpus) {
            CHECK(gpu.weights_cache_total_pages == cache_size) 
                << "Expect same cache size on all GPUs";
        }
    }
    std::cout << "----------------Checking if all workers have the same models in RAM" << std::endl;
    for (auto &p : state.workers[0].models) {
        unsigned model_id = p.first;
        for (auto &worker : state.workers) {
            CHECK(worker.models.find(model_id) != worker.models.end()) 
                << "Inconsistent models across workers";
        }
    }
}

void Scheduler::print_status() {
    unsigned model_pages = 0;
    for (auto &model : models) {
        model_pages += model->num_weights_pages;
    }

    unsigned gpu_pages = 0;
    for (auto &gpu : gpus) {
        gpu_pages += gpu->pages;
    }


    std::cout << "Total GPU capacity " << gpu_pages << " pages (" << gpu_pages/gpus.size() << " per GPU)." << std::endl
              << "Total model pages " << model_pages << " (" << (100*model_pages / gpu_pages) << "% oversubscription)." << std::endl;
}

void Scheduler::initialize_models(ClockworkState &state) {
    models.resize(state.workers[0].models.size(), nullptr);

    for (auto &p : state.workers[0].models) {
        auto &model = p.second;

        if (model.id >= models.size()) {
            models.resize(model.id+1, nullptr);
        }

        models[model.id] = new Model(this, model);  // this also creates model queues
    }

    std::cout << "Created " << models.size() << " models" << std::endl;
}

void Scheduler::initialize_gpus(std::vector<network::controller::WorkerConnection*> workers,
                ClockworkState &state) 
{
    unsigned total_pages = 0;
    unsigned workers_remaining = state.workers.size();
    unsigned gpus_remaining = max_gpus;
    for (WorkerState &worker : state.workers) {
        int num_gpus = std::min((unsigned) worker.gpus.size(), gpus_remaining / workers_remaining);
        for (unsigned i = 0; i < num_gpus; i++) {
            GPUState &gpustate = worker.gpus[i];
            GPU* gpu = new GPU(
                gpus.size(),
                this,
                workers[worker.id],
                worker.id,
                gpustate.id,
                gpustate.weights_cache_total_pages
            );
            gpus.push_back(gpu);

            total_pages += gpu->pages;
        }
        gpus_remaining -= num_gpus;
        workers_remaining--;
    }
    std::cout << "Created " << gpus.size() << " GPUs on " << state.workers.size() << " Workers" << std::endl;
}

//This creates/initializes <ModelInstance>instances for each combination of model/GPU
void Scheduler::initialize_model_instances() {
    for (auto &gpu : gpus) {
        gpu->instances.resize(models.size(), nullptr);
    }
    for (auto &model : models) {
        model->instances.resize(gpus.size(), nullptr);
    }

    for (unsigned i = 0; i < gpus.size(); i++) {
        GPU* gpu = gpus[i];
        for (unsigned j = 0; j < models.size(); j++) {
            Model* model = models[j];
            ModelInstance* instance = new ModelInstance(gpu, model);
            model->instances[i] = instance;
            gpu->instances[j] = instance;
        }
    }

    tracker = new LoadTracker(gpus.size(), models.size(), default_slo);
    // TODO(neolou): lines under are all manual fixes
    tracker->detach(tracker->models[0]);
    tracker->addGPU(tracker->models[0], tracker->gpus[0]);
    tracker->addGPUcomplete(tracker->models[0], tracker->gpus[0]);
    models[0]->instances[0]->loaded = true;
    gpus[0]->instances[0]->loaded = true;
    tracker->attach(tracker->gpus[0]);
    // end of manual fixes
    for (auto model : models) {
        model->tracker = tracker->newModelTracker(model->id);
    }
}


void networkPrintThread(std::vector<network::controller::WorkerConnection*> workers) {
    uint64_t last_print = util::now();
    uint64_t print_interval_nanos = 1000000000UL * 10;

    network::connection_stats previous_stats;
    while (true) {
        uint64_t now = util::now();
        if (last_print + print_interval_nanos > now) {
            usleep(100000);
            continue;
        }

        network::connection_stats stats;
        for (auto &worker : workers) {
            stats += worker->stats;
        }
        stats -= previous_stats;
        previous_stats = stats;

        float duration = (now - last_print) / 1000000000.0;
        stats /= duration;

        std::stringstream msg;
        msg << std::fixed << std::setprecision(1);
        msg << "Network->Workers: ";
        msg << (stats.bytes_sent / (1024*1024.0)) << "MB/s ";
        msg << "(" << stats.messages_sent << " msgs) snd, ";
        msg << (stats.bytes_received / (1024*1024.0)) << "MB/s ";
        msg << "(" << stats.messages_received << " msgs) rcv, ";
        msg << std::endl;

        std::cout << msg.str();

        last_print = now;
    }
}

void Scheduler::run_gpu_stats_printer_thread() {
    uint64_t print_every = 2500000000UL;
    uint64_t last_print = util::now();
    while (true) {
        uint64_t now = util::now();
        if (print_every + last_print <= now) {
            last_print = now;

            std::stringstream s;
            for (auto &gpu : gpus) {
                s << gpu->stats() << std::endl;
            }
            s << "Requests: " << request_count.exchange(0) << std::endl;
            std::cout << s.str();
        }

        usleep(10000);
    }
}

void Scheduler::initialize_network(std::vector<network::controller::WorkerConnection*> workers) {
    auto transmitComplete = [this]() {
        this->network->sendComplete();
    };
    auto transmitError = [this](uint64_t timeout_at, std::shared_ptr<workerapi::Result> result) {
        network_timeout_queue.push({timeout_at, result});
    };

    this->network = new NetworkExecutor(network_concurrency, transmitError);

    for (auto worker : workers) {
        worker->setTransmitCallback(transmitComplete);
    }
}

/**
 * @brief Called when model loading has completed
 * 
 * This function performs several tasks after model loading has completed. It validates the workers' state,
 * creates/initializes models on workers, initializes GPUs, model instances, and the network.
 * It then creates and starts various threads for different tasks like printing stats, processing requests,
 * handling trackers, processing results, running inference, and loading data.
 *
 * @param workers A vector of worker connections
 * @param state The clockwork state
 *
 * @throws None
 */
// Called when model loading has completed
void Scheduler::start(std::vector<network::controller::WorkerConnection*> workers,
                    ClockworkState &state) 
{
    validate_clockwork_state(state);
    initialize_models(state);
    initialize_gpus(workers, state);
    initialize_model_instances();
    initialize_network(workers);

    print_status();
    
    // Create and start the printer threads
    this->printer = ControllerActionTelemetry::log_and_summarize(actions_filename, print_interval);
    network_printer = std::thread(&networkPrintThread, workers);  // Print network stats
    threading::initLoggerThread(network_printer);

    if (print_scheduler_stats) {  // false by default
        this->stats_printer = std::thread(&Scheduler::run_gpu_stats_printer_thread, this);
        threading::initLoggerThread(stats_printer);  // Print GPU stats
    }

    uint64_t num_admission_threads = 2; // 2
    for (int i = 0; i < num_admission_threads; i++) {
        admission_threads.push_back(std::thread(&Scheduler::run_admission_thread, this));
        threading::initHighPriorityThread(admission_threads[i]);  // Process/handle requests, adding them to the model queues, dropping ones that timed out
    }

    uint64_t num_tracker_threads = 1;
    for (int i = 0; i < num_tracker_threads; i++) {
        tracker_threads.push_back(std::thread(&Scheduler::run_tracker_thread, this));  // Processes the trackers of all the 'stale' models
        threading::initHighPriorityThread(tracker_threads[i]);
    }

    uint64_t num_results_threads = 2; // 2
    for (int i = 0; i < num_results_threads; i++) {
        results_threads.push_back(std::thread(&Scheduler::run_results_thread, this));  // Processes the results from the result_queue
        threading::initHighPriorityThread(results_threads[i]);
    }

    int num_infer_threads = 5; // 5
    for (unsigned i = 0; i < num_infer_threads; i++) {
        infer_threads.push_back(std::thread(&Scheduler::run_infer_thread, this, i));
        threading::initHighPriorityThread(infer_threads[i]);
    }
    for (auto gpu : gpus) {
        to_infer.push(gpu);
    }

    int num_load_threads = 5; // 5
    for (unsigned i = 0; i < num_load_threads; i++) {
        load_threads.push_back(std::thread(&Scheduler::run_load_thread, this, i));
        threading::initHighPriorityThread(load_threads[i]);
    }
    for (auto gpu : gpus) {
        to_load.push(gpu);
    }
}

struct tracker_request {
    int model_id;
    uint64_t size;
    uint64_t start_exec_by;
    uint64_t start_loadweights_by;
};

/**
 * Handles a request by setting up its model, Service Level Objective (SLO), demand (execution/load time), and submits the request to the queue for the model.
 *
 * @param request The request object containing model information
 *
 * @throws None
 */
void Scheduler::handle_request(Request &request) {
    int model_id = request->request.model_id;
    Model* model = models[model_id];
    request->set_model(model);
    // request->set_slo(default_slo);
    request->set_slo(request->request.slo);

    // request->demand = model->tracker->addRequest(  // Calculate the demand for the request
    //     model->estimate(1), request->exec_slo, request->weights_slo);
    model->invalidate_tracker();
    request->model->enqueue(request);
}

void Scheduler::add_callback(uint64_t action_id, Callback callback) {
    auto pair = std::make_pair(action_id, callback);

    tbb::queuing_mutex::scoped_lock lock(callbacks_mutex);
    callbacks.insert(pair);
}

/**
* Handles the result of a worker action by calling its corresponding callback.
*
* @param result A shared pointer to the worker action result.
*
* @throws None
*/
void Scheduler::handle_result(std::shared_ptr<workerapi::Result> &result) {
    Callback callback;
    {
        tbb::queuing_mutex::scoped_lock lock(callbacks_mutex);
        auto it = callbacks.find(result->id);
        CHECK(it != callbacks.end()) 
            << "Received result for non-existent action " << result->str();

        callback = it->second;
        callbacks.erase(it);
    }
    callback(result);
}

/**
 * Runs the admission thread of the Scheduler.
 *
 * This function continuously pops requests from the request queue and handles them.
 * It also drops requests that have timed out.
 *
 * @return void
 *
 * @throws None
 */
void Scheduler::run_admission_thread() {
    // Process this many requests per iteration
    int max_requests = 50;
    std::vector<Request> requests;
    requests.reserve(max_requests);

    std::priority_queue<Request, std::deque<Request>, RequestImpl::DeadlineComparator> timeout_queue;

    int i = 0;
    while (true) {
        bool active = false;

        // Pop a request
        Request request;
        if (request_queue.try_pop(request)) {
            // Immediately drop requests to invalid models
            unsigned model_id = request->request.model_id;
            if (model_id > models.size() || models[model_id] == nullptr) {
                request->set_error(clockworkError, "Invalid model ID");
                CHECK(!request->complete(util::now(), -1)) << "Erroneous request should not be successful";
            } else {
                handle_request(request);  // Handles the request
                timeout_queue.push(request);
            }
            active = true;
            i++;
        }

        // Drop any timed out requests
        uint64_t now = util::now();
        while (!timeout_queue.empty()) {
            auto &request = timeout_queue.top();

            if (request->deadline > now) break;

            request->finalize();
            timeout_queue.pop();
            active = true;
            i++;
        }

        if (!active || i >= 100) {
            usleep(10);
            i = 0;
        }
    }
}

/**
 * Runs the results thread.
 *
 * This function continuously pops a result from the result queue and handles it.
 * It also checks for timed out results and handles them.
 *
 * @throws None
 */
void Scheduler::run_results_thread() {
    bool should_timeout = false;
    TimeoutResult next_timeout;

    int i = 0;
    while (true) {
        bool active = false;

        std::shared_ptr<workerapi::Result> result;
        if (result_queue.try_pop(result)) {
            handle_result(result);
            active = true;
            i++;
        }

        if (!should_timeout) {
            should_timeout = network_timeout_queue.try_pop(next_timeout);
        }

        if (should_timeout) {
            if (next_timeout.timeout_at <= util::now()) {
                handle_result(next_timeout.result);
                should_timeout = false;
                active = true;
                i++;
            }
        }

        if (!active || i >= 100) {
            usleep(10);
            i = 0;
        }
    }
}

void Scheduler::run_infer_thread(int id) {
    std::stringstream msg;
    msg << "GPU infer thread [" << id << "] started" << std::endl;
    std::cout << msg.str();

    int i = 0;
    int n_gpus = gpus.size();
    while (true) {
        uint64_t i = (next_infer++) % n_gpus;
        bool active = gpus[i]->schedule_infer();
        usleep(10);
    }
}

void Scheduler::run_load_thread(int id) {
    std::stringstream msg;
    msg << "GPU load thread [" << id << "] started" << std::endl;
    std::cout << msg.str();

    int inactive = 0;
    int n_gpus = gpus.size();
    while (true) {
        uint64_t i = (next_load++) % n_gpus;
        bool active = gpus[i]->schedule_load();
        usleep(10);
    }

}
/**
 * This function runs the tracker thread. It processes models by popping them from 'stale' into a vector of 'models',
 * processes the tracker of each model in 'models', and then resets the tracker. 
 *
 * @return void
 *
 * @throws None
 */
void Scheduler::run_tracker_thread() {
    std::cout << "Tracker thread running\n";
    std::vector<Model*> models;
    while (true) {
        Model* model;
        while (stale.try_pop(model)) {
            models.push_back(model);
        }

        if (models.size() > 0) {
            tbb::queuing_mutex::scoped_lock lock(tracker->mutex);

            for (auto &model : models) {
                tracker->process(model->tracker);
                model->reset_tracker();
            }
        }

        models.clear();

        usleep(10);
    }
}

// The actual scheduler interface implementation, invoked by worker network thread
void Scheduler::resultFromWorker(std::shared_ptr<workerapi::Result> result)
{
    if (print_debug) std::cout << ("Worker  --> " + result->str() + "\n");

    result->result_received = util::now();
    result_queue.push(result);
}

// The actual scheduler interface implementation, invoked by client network thread
void Scheduler::clientInfer(clientapi::InferenceRequest &request, 
    std::function<void(clientapi::InferenceResponse&)> callback)
{
    if (print_debug) std::cout << ("Client  --> " + request.str() + "\n");

    request_queue.push(std::make_shared<RequestImpl>(this, request, callback));
    request_count++;
}

Scheduler::NetworkExecutor::NetworkExecutor(unsigned concurrency, 
    std::function<void(uint64_t, std::shared_ptr<workerapi::Result>)> error_callback) : 
idle(concurrency), error_callback(error_callback) {}

void Scheduler::NetworkExecutor::send(network::controller::WorkerConnection* worker, 
          std::shared_ptr<workerapi::Action> action,
          uint64_t start_send_by,
          uint64_t send_error_at) {

    NetworkAction toSend;
    {
        tbb::queuing_mutex::scoped_lock lock(mutex);

        pending.push_back({worker, action, start_send_by, send_error_at});

        if (idle == 0) return;

        if (!next(toSend)) return;

        idle--;
    }

    toSend.worker->sendAction(toSend.action);            
}

void Scheduler::NetworkExecutor::sendComplete() {
    NetworkAction toSend;
    {
        tbb::queuing_mutex::scoped_lock lock(mutex);
        if (!next(toSend)) {
            idle++;
            return;
        }
    }

    toSend.worker->sendAction(toSend.action);
}

bool Scheduler::NetworkExecutor::next(NetworkAction &toSend) {
    uint64_t now = util::now();
    while (pending.size() > 0) {
        toSend = pending.front();
        pending.pop_front();

        if (toSend.start_send_by >= now) {
            return true;
        }

        auto action = toSend.action;
        auto result = std::make_shared<workerapi::ErrorResult>();
        result->id = action->id;
        result->action_type = action->action_type;
        result->status = networkSendTooLate;
        result->action_received = now;
        result->result_sent = now;
        result->result_received = now;
        result->message = "Could not send action to worker in time";

        error_callback(toSend.send_error_at, result);

    }
    return false;
}

}
}
}