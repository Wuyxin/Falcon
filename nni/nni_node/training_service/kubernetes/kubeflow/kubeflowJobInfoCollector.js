"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.KubeflowJobInfoCollector = void 0;
const kubernetesJobInfoCollector_1 = require("../kubernetesJobInfoCollector");
class KubeflowJobInfoCollector extends kubernetesJobInfoCollector_1.KubernetesJobInfoCollector {
    constructor(jobMap) {
        super(jobMap);
    }
    async retrieveSingleTrialJobInfo(kubernetesCRDClient, kubernetesTrialJob) {
        if (!this.statusesNeedToCheck.includes(kubernetesTrialJob.status)) {
            return Promise.resolve();
        }
        if (kubernetesCRDClient === undefined) {
            return Promise.reject('kubernetesCRDClient is undefined');
        }
        let kubernetesJobInfo;
        try {
            kubernetesJobInfo = await kubernetesCRDClient.getKubernetesJob(kubernetesTrialJob.kubernetesJobName);
        }
        catch (error) {
            this.log.error(`Get job ${kubernetesTrialJob.kubernetesJobName} info failed, error is ${error}`);
            return Promise.resolve();
        }
        if (kubernetesJobInfo.status && kubernetesJobInfo.status.conditions) {
            const latestCondition = kubernetesJobInfo.status.conditions[kubernetesJobInfo.status.conditions.length - 1];
            const tfJobType = latestCondition.type;
            switch (tfJobType) {
                case 'Created':
                    kubernetesTrialJob.status = 'WAITING';
                    kubernetesTrialJob.startTime = Date.parse(latestCondition.lastUpdateTime);
                    break;
                case 'Running':
                    kubernetesTrialJob.status = 'RUNNING';
                    if (kubernetesTrialJob.startTime === undefined) {
                        kubernetesTrialJob.startTime = Date.parse(latestCondition.lastUpdateTime);
                    }
                    break;
                case 'Failed':
                    kubernetesTrialJob.status = 'FAILED';
                    kubernetesTrialJob.endTime = Date.parse(latestCondition.lastUpdateTime);
                    break;
                case 'Succeeded':
                    kubernetesTrialJob.status = 'SUCCEEDED';
                    kubernetesTrialJob.endTime = Date.parse(latestCondition.lastUpdateTime);
                    break;
                default:
            }
        }
        return Promise.resolve();
    }
}
exports.KubeflowJobInfoCollector = KubeflowJobInfoCollector;
