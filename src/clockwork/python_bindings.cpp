#include "clockwork/api/worker_api.h"
#include "clockwork/controller/infer5/infer5_scheduler.h"
#include "clockwork/controller/scheduler.h"
#include <cstdint>
#include <python3.7m/Python.h>
#include <python3.7m/object.h>
#include <memory>
#include <sys/types.h>
#include <vector>
#include <nvml.h>

#define SCHED_T clockwork::scheduler::infer5::Scheduler

// This file has code for binding Python-C++ with ctypes

/********************* Helper functions/classes *************************/

class AdmissionThreadMemory {
    public:
        AdmissionThreadMemory() {
            requests.reserve(max_requests);
        }
        bool admission_first_loop = true;
        int max_requests = 50;
        std::vector<SCHED_T::Request> requests;
        std::priority_queue<SCHED_T::Request, 
                            std::deque<SCHED_T::Request>,
                            SCHED_T::RequestImpl::DeadlineComparator> timeout_queue;
        int dropped = 0; // counts number of dropped reqs
        bool active;
};

static void checkPyObjType(PyObject* obj) {
    PyObject* typeObj = PyObject_Type(obj);
    if (!typeObj) {
        std::cout << "Failed to get type of object" << std::endl;
    }
    PyObject* strObj = PyObject_Str(typeObj);
    if (!strObj) {
        std::cout << "Failed to convert type object to string" << std::endl;
        Py_DECREF(typeObj);
    }
    const char* typeStr = PyUnicode_AsUTF8(strObj);
    if (typeStr) {
        std::cout << "Type of obj: " << typeStr << std::endl;
    } else {
        std::cout << "Failed to convert type string to char array" << std::endl;
    }
    Py_DECREF(strObj); // Clean up strObj reference
    Py_DECREF(typeObj); // Clean up typeObj reference
}

static std::vector<unsigned> convertPyListToUnsignedVector(PyObject* pyList) {
    std::vector<unsigned> resultVector;
    if (!PyList_Check(pyList)) {
        std::cerr << "Provided PyObject is not a list." << std::endl;
        return resultVector; // Return an empty vector
    }

    Py_ssize_t listSize = PyList_Size(pyList);
    for (Py_ssize_t i = 0; i < listSize; ++i) {
        PyObject* item = PyList_GetItem(pyList, i); // Borrowed reference
        if (!item) {
            // Handle error
            std::cerr << "Failed to get item from list at index " << i << std::endl;
            break;
        }
        unsigned itemValue = PyLong_AsUnsignedLong(item);
        if (PyErr_Occurred()) {
            // Handle conversion error
            std::cerr << "Failed to convert list item to unsigned long at index " << i << std::endl;
            PyErr_Clear(); // Clear the error indicator
            continue; // Skip this item or break as needed
        }
        resultVector.push_back(itemValue);
    }

    return resultVector;
}

/***************** Callable C++ functions from Python bindings library *******************/
extern "C" {
    
    // Returns the current time in nanoseconds, clockwork style
    std::uint64_t getClockworkTimeNow() {
        auto time_now = util::now();
        return time_now;  // in nanoseconds
    }

    // Get GPU clock speed in MHz
    unsigned int getClockSpeed(nvmlDevice_t gpu_device) {
        unsigned int clock_speed;
        auto status = nvmlDeviceGetClockInfo(gpu_device, NVML_CLOCK_SM, &clock_speed);
        if (status != NVML_SUCCESS) {
            std::cerr << "Failed to get GPU clock speed" << std::endl;
            return 0;
        }
        std::cout << "GPU clock speed: " << clock_speed << " MHz" << std::endl;
        return clock_speed;
    }
    
    void *createScheduler() {
        // Based on src/controller.cpp: int main(int argc, char *argv[]) 
        bool generate_inputs = false;
        int max_gpus = 100;
        uint64_t schedule_ahead = (generate_inputs ? 15000000UL : 10000000UL);
        uint64_t default_slo = 100000000UL;
        uint64_t max_exec_time = 250000000UL;
        int max_batch_size = 8;
        std::string actions_filename = util::get_controller_log_dir() + "/clockwork_action_log.tsv";
        SCHED_T* scheduler = new SCHED_T(
            default_slo,
            schedule_ahead,
            schedule_ahead,
            generate_inputs,
            max_gpus,
            max_exec_time,
            max_batch_size,
            actions_filename
        );
        std::cout << "Created Clockwork Scheduler" << std::endl;
        return scheduler;
    }

    // Start the Clockwork scheduler, somewhat Based on: src/clockwork/controller/infer5/infer5_scheduler.cpp: void Scheduler::start(...)
    void startScheduler(SCHED_T *scheduler, PyObject *model_node_dict, PyObject *gpus_dict, PyObject *loaded_model_placements_list, int max_message_length) {
        /*  C++ type definitions
        struct ClockworkState {
            size_t page_size;
            std::vector<WorkerState> workers;
            std::string str();
        };
        struct WorkerState {
            unsigned id;
            std::vector<GPUState> gpus;
            std::map<unsigned, BatchedModelState> models;
            std::string str();
        };
        struct BatchedModelState {
            unsigned id;
            std::string model_path;
            size_t input_size;
            size_t output_size;
            size_t weights_size; // Total size or size in pages?
            unsigned num_weights_pages;
            uint64_t weights_transfer_duration; // in nanosec?
            std::vector<unsigned> supported_batch_sizes;
            std::map<unsigned, uint64_t> exec_duration; // map of batch size to exec duration
            std::string str();
        };
        struct GPUState {
            unsigned id;
            size_t weights_cache_size;
            unsigned weights_cache_total_pages;   // Number of pages in GPU weights cache
            std::vector<unsigned> loaded_models;  // Models loaded into GPU memory
            std::string str();
        };
        */
        std::cout << "Start Clockwork Scheduler" << std::endl;

        // Create ClockworkState
        ClockworkState state;
        state.page_size = 16777216L;  // from config/default.cfg
        state.workers = {};
        std::cout << "Initial Clockwork state: " << state.str() << std::endl;

        // Populate ClockworkState workers
        // Based on: void QueryWorkerStage::populate_worker_state(...)

        // Generate WorkerState
        PyObject *model_id, *node_local_id_list;
        Py_ssize_t pos = 0;
        bool first_loop = true;
        while (PyDict_Next(model_node_dict, &pos, &model_id, &node_local_id_list)) { // key: model_id, value: node_id
            
            // Create batched_model_state (for worker_state)
            // Based on: void QueryWorkerStage::populate_model_state(...)

            // Example of config for resnet50v2 model, based on file: config/model.measurements.example
            /* These numbers specifically come from another repository "clockwork-modelzoo-volta": clockwork-modelzoo-volta/resnet50_v2/model.measurements
            weights = 8372000L;  // weights_transfer_duration
            b1 = 3322000L;  // infer exec duration for batch size 1
            b2 = 5271340L;  // infer exec duration for batch size 2
            b4 = 7495000L;  // etc.
            b8 = 12439000L;
            b16 = 21728879L;
            */
            
            unsigned model_id_cpp = PyLong_AsUnsignedLong(model_id);
            std::vector<unsigned> node_local_id_list_cpp = convertPyListToUnsignedVector(node_local_id_list);

            // Right now, create all models with same parameters, but change this in the future
            BatchedModelState batched_model_state;
            batched_model_state.id = model_id_cpp;
            batched_model_state.weights_transfer_duration = 8372000L; // hard code for now, need to profile to know whats the value for our infrastructure
            // TODO(neolou): need to calculate weight of first model loaded in mem for free_pages
            batched_model_state.num_weights_pages = 320;  // hard code, random value i chose, need to profile to know whats the value for our infrastructure
            std::vector<unsigned> supported_batch_sizes = {1, 2, 4, 8}; // hard code for now
            std::vector<uint64_t> batch_size_exec_times_nanos = {3322000L, 5271340L, 7495000L, 12439000L}; // hard code for now, need to profile to know whats the value for our infrastructure
            batched_model_state.supported_batch_sizes = supported_batch_sizes; 
            for (unsigned i = 0; i < supported_batch_sizes.size(); i++) {
                auto batch_size = supported_batch_sizes[i];
                auto duration = batch_size_exec_times_nanos[i];
                batched_model_state.exec_duration[batch_size] = duration;
            }
            
            // If first loop, create new worker_state for each node
            if (first_loop) {
                for (auto node_local_id : node_local_id_list_cpp) {
                    WorkerState worker_state;
                    worker_state.id = node_local_id;
                    worker_state.models.insert({model_id_cpp,batched_model_state});
                    state.workers.push_back(worker_state);
                }
                first_loop = false;
            }
            else {
                for (auto &worker_state : state.workers) {
                    for (auto node_local_id : node_local_id_list_cpp) {
                        // If worker_state exists, add model to it
                        if (worker_state.id == node_local_id) {
                            worker_state.models.insert({model_id_cpp, batched_model_state});
                            break;
                        }
                    }
                }
            }
        }
        Py_XDECREF(model_id);  // dereference pyobj created
        Py_XDECREF(node_local_id_list);  // dereference pyobj created

        // Now, populate gpus for each worker_state
        // Based on: void QueryWorkerStage::populate_gpu_state(...)
        
        // Generate GPUState
        PyObject *gpu_global_id, *gpu_obj;
        pos = 0;
        while (PyDict_Next(gpus_dict, &pos, &gpu_global_id, &gpu_obj)) { // key: gpu_global_id, value: gpu_obj
            // Create and populate gpu_state
            GPUState gpu_state;
            unsigned gpu_id = PyLong_AsUnsignedLong(PyObject_GetAttrString(gpu_obj, "local_id"));
            gpu_state.id = gpu_id;
            gpu_state.weights_cache_size = 10737418240L;  // from config/default.cfg
            gpu_state.weights_cache_total_pages = 10737418240L/16777216L;  // total_pages = cache_size / page_size (values from config/default.cfg)

            // Populate gpu_state.loaded_models
            Py_ssize_t list_size = PyList_Size(loaded_model_placements_list);
            for (Py_ssize_t i = 0; i < list_size; ++i) {  // Iterate over model placements
                PyObject *model_p = PyList_GetItem(loaded_model_placements_list, i); // get model placement
                PyObject *model_p_gpu = PyObject_GetAttrString(model_p, "gpu"); // get gpu of model placement
                if (gpu_id == PyLong_AsUnsignedLong(PyObject_GetAttrString(model_p_gpu, "local_id"))) {
                    // if gpu id matches, add model id to loaded models of gpu
                    PyObject *model = PyObject_GetAttrString(model_p, "model");
                    unsigned model_id = PyLong_AsUnsignedLong(PyObject_GetAttrString(model, "id"));
                    gpu_state.loaded_models.push_back(model_id);
                }
            }

            // Insert gpu_state into worker_state
            PyObject *node = PyObject_GetAttrString(gpu_obj, "node");
            auto node_id = PyLong_AsUnsignedLong(PyObject_GetAttrString(node, "local_id"));
            for (auto &worker_state : state.workers) {
                if (worker_state.id == node_id) {
                    worker_state.gpus.push_back(gpu_state);
                    break;
                }
            }
        }
        Py_XDECREF(gpu_global_id);  // dereference pyobj created
        Py_XDECREF(gpu_obj);  // dereference pyobj created
        
        // scheduler->validate_clockwork_state(state);

        std::cout << "Initialized models, gpus, and model instances" << std::endl;
        scheduler->initialize_models(state); // populate "models" field
        std::vector<network::controller::WorkerConnection *> workers;
        for (unsigned i = 0; i < state.workers.size(); i++) {
            network::controller::WorkerConnection* worker;
            workers.push_back(worker);
        }
        scheduler->initialize_gpus(workers, state);
        scheduler->initialize_model_instances();  // TODO(neolou): manual fix to make load work

        //scheduler->initialize_network(workers);
        
        // Create grpc channel and stub for each gpu
        std::cout << "Create gRPC channel and stub for each of the " << scheduler->gpus.size() << " GPUs" << std::endl;
        grpc::ChannelArguments args;
        args.SetMaxReceiveMessageSize(max_message_length); // increase channel's max receive message size
        args.SetMaxSendMessageSize(max_message_length);  // increase channel's max send message size
        for (int i = 0; i < scheduler->gpus.size(); i++) {
            auto cur_gpu = scheduler->gpus[i];
            cur_gpu->channel_ = grpc::CreateCustomChannel("localhost:50051", grpc::InsecureChannelCredentials(), args);
            cur_gpu->stub_ = cluster_comm::NodeController::NewStub(cur_gpu->channel_);
        }
        std::cout << "Finished starting Clockwork Scheduler, final Clockwork State: " << state.str() << std::endl;
    }

    // Convert orion request data to clockwork request and insert into clockwork scheduler
     void insertOrionRequestInClockwork(SCHED_T *scheduler, char *req_uuid, int req_local_id, int client_id, int model_id, int slo,
                                        char *tensor_bytes, int tensor_bytes_size, int *tensor_shape, int tensor_shape_size) {
        /*
        ------------------------------- Request data format -------------------------------
        struct InferenceRequest {
            RequestHeader header; # struct RequestHeader {int user_id; int user_request_id;};
            int model_id;
            int batch_size;
            size_t input_size; # unsigned long
            void* input;
            uint64_t deadline = 0;
            float slo_factor;

            // Not sent over the network; used by controller
            uint64_t arrival = 0;

            std::string str();
        };
        ----------------------------- Python tensor format -----------------------------
        torch.rand([1, 3, 224, 224])
        */
        
        // Based on src/clockwork/client.cpp: void ModelImpl::infer(...)

        // Copy tensor_bytes into new tensor_bytes
        char* new_tensor_bytes = new char[tensor_bytes_size];
        std::memcpy(new_tensor_bytes, tensor_bytes, tensor_bytes_size);
        // Create vector for tensor_shape
        std::vector<int> tensor_shape_vec(tensor_shape, tensor_shape + tensor_shape_size);

        // Create InferenceRequest
        auto request = new clientapi::InferenceRequest();
	    request->header.user_id = client_id;
	    request->header.user_request_id = req_local_id;
        request->uuid = std::string(req_uuid);
	    request->model_id = model_id;
	    request->batch_size = 1;
        request->slo_factor = 0;
        request->input = new_tensor_bytes;
	    request->input_size = tensor_bytes_size;
        request->input_shape = tensor_shape_vec;
        request->slo = slo;
         // create blank callback (can't find what it was originally supposed to be... but probably some networking thing)
        std::function<void (clientapi::InferenceResponse &)> callback = [] (clientapi::InferenceResponse &) {};
        if (request->arrival == 0) {
		    request->arrival = util::now();
	    }
        scheduler->clientInfer(*request, callback);
    }

    // Convert orion inference result into clockwork result and insert into clockwork scheduler
    void insertOrionInferResultInClockwork(
        SCHED_T *scheduler, int infer_id, uint64_t exec_start, uint64_t exec_end,
        uint64_t time_message_sent, uint64_t time_message_received, int gpu_clock_speed_before, int gpu_clock_speed_after) {
        /*
        ------------------------------- Result data format -------------------------------
        tbb::concurrent_queue<std::shared_ptr<workerapi::Result>> result_queue;
        class InferResult : public Result {
            public:
                Timing copy_input;
                Timing exec;
                Timing copy_output;
                int output_size;
                char* output;
                unsigned gpu_id;
                unsigned gpu_clock_before;
                unsigned gpu_clock;
                
                virtual std::string str();
        };
        */

        // TODO(neolou): check if this way of initializing "InferResult" are faithful to clockwork policy and that all the important info is captured

        // Create InferenceRequest
        // Based on src/clockwork/action.cpp: void InferAction::handle_completion(char* output)

        std::shared_ptr<workerapi::InferResult> result = std::make_shared<workerapi::InferResult>();
        /* Variables needed to proceed:
	    result->id;
        result->exec.end;
        result->exec.duration;
        result->gpu_clock;
        result->gpu_clock_before;
        */

        result->id = infer_id;
        result->action_type = workerapi::inferAction;
        result->status = actionSuccess;
        result->output_size = 0;  // we don't care about output because we have it in orion
        result->output = nullptr;  // we don't care about output because we have it in orion

        // Ignore copy_input and copy_output for now
        // extract_timing_async(&result->copy_input, copy_input->telemetry);
        // extract_timing_async(&result->copy_output, copy_output->telemetry);
        result->exec.begin = exec_start;
        result->exec.end = exec_end;
        result->exec.duration = exec_end - exec_start;

        // result->gpu_id = action->gpu_id;  // i think this is unecessary?
        result->gpu_clock_before = gpu_clock_speed_before; // GPU clock speed before infer, in MHz
        result->gpu_clock = gpu_clock_speed_after; // cGPU lock speed after infer, in MHz

        // Implementation of this section (synchronize, clock_delta) Based on src/clockwork/network/network.cpp: void message_receiver::handle_header_read(...)

        // local_delta_ = min of window of time between message sent/read = rx_begin_ - pre_header[4] = [when message read] - [when message sent]
        // remote_delta_ = min of window of time of previous local_delta_'s
        int64_t prev_local_delta = scheduler->scheduler_message_handler.local_delta_;
        int64_t new_local_delta = time_message_received - time_message_sent;
        scheduler->scheduler_message_handler.synchronize(new_local_delta, prev_local_delta);
        auto clock_delta = scheduler->scheduler_message_handler.estimate_clock_delta();

        // Based on src/clockwork/worker.cpp: void Infer::success(std::shared_ptr<workerapi::InferResult> result)

        // result->copy_input.begin = adjust_timestamp(result->copy_input.begin, -action->clock_delta);
        // result->copy_input.end = adjust_timestamp(result->copy_input.end, -action->clock_delta);
        result->exec.begin = adjust_timestamp(result->exec.begin, -clock_delta);
        result->exec.end = adjust_timestamp(result->exec.end, -clock_delta);
        // result->copy_output.begin = adjust_timestamp(result->copy_output.begin, -action->clock_delta);
        // result->copy_output.end = adjust_timestamp(result->copy_output.end, -action->clock_delta);
        result->action_received = adjust_timestamp(time_message_received, -clock_delta);
        result->clock_delta = clock_delta;


        scheduler->resultFromWorker(result);
    }

    // Convert orion load weights result into clockwork result and insert into clockwork scheduler
    void insertOrionLoadResultInClockwork(SCHED_T *scheduler, int load_id, uint64_t load_start, uint64_t load_end) {
        /*
        ------------------------------- Result data format -------------------------------
        tbb::concurrent_queue<std::shared_ptr<workerapi::Result>> result_queue;
        class LoadWeightsResult : public Result, public Timing {
            public:
                virtual std::string str();
        };
        */

        // TODO(neolou): check if this way of initializing "LoadWeightsResult" are faithful to clockwork policy and that all the important info is captured

        std::shared_ptr<workerapi::LoadWeightsResult> result = std::make_shared<workerapi::LoadWeightsResult>();

        /* Variables needed to proceed:
        result->id
        result->end
        result->duration
        */
        // ControllerActionTelemetry::set(std::shared_ptr<workerapi::LoadWeightsResult> &result)

        result->id = load_id;
        result->action_type = workerapi::loadWeightsAction;
        result->status = actionSuccess;
        result->end = load_end;
        result->duration = load_end - load_start;

        // Callback function: void Scheduler::GPU::load_result(LoadWeightsAction* action, std::shared_ptr<workerapi::Result> &result)
        // Check what else in result needs to be populated before adding to scheduler

        scheduler->resultFromWorker(result);
    }

    // Convert orion evict weights result into clockwork result and insert into clockwork scheduler
    void insertOrionEvictResultInClockwork(SCHED_T *scheduler, int evict_id) {
        /*
        ------------------------------- Result data format -------------------------------
        tbb::concurrent_queue<std::shared_ptr<workerapi::Result>> result_queue;
        class LoadWeightsResult : public Result, public Timing {
            public:
                virtual std::string str();
        };
        */

        std::shared_ptr<workerapi::EvictWeightsResult> result = std::make_shared<workerapi::EvictWeightsResult>();

        result->id = evict_id;
        result->action_type = workerapi::evictWeightsAction;
        result->status = actionSuccess;

        // Callback function: void Scheduler::GPU::evict_result(EvictWeightsAction* action, std::shared_ptr<workerapi::Result> &result)
        
        scheduler->resultFromWorker(result);
    }



    AdmissionThreadMemory *initAdmissionThread() {
        return new AdmissionThreadMemory();
    }

    bool isAdmissionThreadActive(AdmissionThreadMemory *admission_thread_memory) {
        return admission_thread_memory->active;
    }

    // Clockwork runs 2 admission threads
    // Based on src/clockwork/controller/infer5/infer5_scheduler.cpp: void Scheduler::run_admission_thread()
    AdmissionThreadMemory *runAdmissionThread(SCHED_T *scheduler, AdmissionThreadMemory *admission_thread_memory) {

        // Process this many requests per iteration
        // int max_requests = 50;
        // std::vector<SCHED_T::Request> requests;
        // requests.reserve(max_requests);
        // std::priority_queue<SCHED_T::Request, 
        //                     std::deque<SCHED_T::Request>,
        //                     SCHED_T::RequestImpl::DeadlineComparator> timeout_queue;
        // int i = 0;

        // while (true) {
        admission_thread_memory->active = false;

        // Pop a request
        SCHED_T::Request request;
        if (scheduler->request_queue.try_pop(request)) {
            // Immediately drop requests to invalid models
            unsigned model_id = request->request.model_id;
            if (model_id > scheduler->models.size() || scheduler->models[model_id] == nullptr) {
                std::cout << "Invalid model ID" << std::endl;
                request->set_error(clockworkError, "Invalid model ID");
                CHECK(!request->complete(util::now(), -1)) << "Erroneous request should not be successful";
            } else {
                scheduler->handle_request(request);
                admission_thread_memory->timeout_queue.push(request);
            }
            admission_thread_memory->active = true;
            // admission_thread_memory->i++;
        }

        // Process timed out requests
        uint64_t now = util::now();
        while (!admission_thread_memory->timeout_queue.empty()) {
            auto &request = admission_thread_memory->timeout_queue.top();
            if (request->deadline > now) break;
            request->finalize();
            admission_thread_memory->timeout_queue.pop();
            admission_thread_memory->active = true;
            // admission_thread_memory->i++;
            admission_thread_memory->dropped++;
        }

        // if (!active || admission_thread_memory->i >= 100) {
        //     std::cout << "Admission thread sleeping" << std::endl;
        //     usleep(10);
        //     admission_thread_memory->i = 0;
        // }
        // }
        return admission_thread_memory;
    }

    // Clockwork runs 5 infer threads
    // Based on src/clockwork/controller/infer5/infer5_scheduler.cpp: void Scheduler::run_infer_thread(int id)
    bool runInferThread(SCHED_T *scheduler) {
        // std::stringstream msg;
        // msg << "GPU infer thread [" << id << "] started" << std::endl;
        // std::cout << msg.str();

        // int i = 0;
        int n_gpus = scheduler->gpus.size();

        // while (true) {
        uint64_t i = (scheduler->next_infer++) % n_gpus;
        bool active = scheduler->gpus[i]->schedule_infer();
        // }
        return active;
    }

    // Clockwork runs 2 results threads
    // Based on src/clockwork/controller/infer5/infer5_scheduler.cpp: void Scheduler::run_results_thread()
    bool runResultThread(SCHED_T *scheduler) {
        // std::cout << "Result thread running\n";
        // bool should_timeout = false;
        // SCHED_T::TimeoutResult next_timeout;

        // int i = 0;
        // while (true) {
        bool active = false;

        std::shared_ptr<workerapi::Result> result;
        if (scheduler->result_queue.try_pop(result)) {
            scheduler->handle_result(result);
            active = true;
            // i++;
        }
        return active;

        // if (!should_timeout) {
        //   should_timeout = scheduler->network_timeout_queue.try_pop(next_timeout);
        // }

        // if (should_timeout) {
        //     if (next_timeout.timeout_at <= util::now()) {
        //         scheduler->handle_result(next_timeout.result);
        //         should_timeout = false;
        //         active = true;
        //         i++;
        //     }
        // }

        // if (!active || i >= 100) {
        //     usleep(10);
        //     i = 0;
        // }
        // }
    }

    // Clockwork runs 5 load threads
    // Based on src/clockwork/controller/infer5/infer5_scheduler.cpp: void Scheduler::run_load_thread(int id)
    void runLoadThread(SCHED_T *scheduler) {
        // std::stringstream msg;
        // msg << "GPU load thread [" << id << "] started" << std::endl;
        // std::cout << msg.str();

        // int inactive = 0;
        int n_gpus = scheduler->gpus.size();
        // while (true) {
        uint64_t i = (scheduler->next_load++) % n_gpus;
        bool active = scheduler->gpus[i]->schedule_load();
        // usleep(10);
        // }

    }

    // Clockwork runs 1 tracker threads
    // Based on src/clockwork/controller/infer5/infer5_scheduler.cpp: void Scheduler::run_tracker_thread()
    void runTrackerThread(SCHED_T *scheduler) {
        // std::cout << "Tracker thread running\n";
        // std::vector<SCHED_T::Model*> models; // replaced by: scheduler->tracker_models
        // while (true) {
        SCHED_T::Model* model;
        while (scheduler->stale.try_pop(model)) {
            scheduler->tracker_models.push_back(model);
        }

        if (scheduler->tracker_models.size() > 0) {
            tbb::queuing_mutex::scoped_lock lock(scheduler->tracker->mutex);

            for (auto &model : scheduler->tracker_models) {
                scheduler->tracker->process(model->tracker);
                model->reset_tracker();
            }
        }

        scheduler->tracker_models.clear();

        // usleep(10); // sleep 10 microseconds
        // }
    }
}