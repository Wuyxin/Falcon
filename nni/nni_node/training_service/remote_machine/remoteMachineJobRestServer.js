"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.RemoteMachineJobRestServer = void 0;
const clusterJobRestServer_1 = require("../common/clusterJobRestServer");
class RemoteMachineJobRestServer extends clusterJobRestServer_1.ClusterJobRestServer {
    remoteMachineTrainingService;
    constructor(remoteMachineTrainingService) {
        super();
        this.remoteMachineTrainingService = remoteMachineTrainingService;
    }
    handleTrialMetrics(jobId, metrics) {
        for (const singleMetric of metrics) {
            this.remoteMachineTrainingService.MetricsEmitter.emit('metric', {
                id: jobId,
                data: singleMetric
            });
        }
    }
}
exports.RemoteMachineJobRestServer = RemoteMachineJobRestServer;
