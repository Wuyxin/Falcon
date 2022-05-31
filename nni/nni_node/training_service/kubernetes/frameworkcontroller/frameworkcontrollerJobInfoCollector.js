"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.FrameworkControllerJobInfoCollector = void 0;
const kubernetesJobInfoCollector_1 = require("../kubernetesJobInfoCollector");
class FrameworkControllerJobInfoCollector extends kubernetesJobInfoCollector_1.KubernetesJobInfoCollector {
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
        if (kubernetesJobInfo.status && kubernetesJobInfo.status.state) {
            const frameworkJobType = kubernetesJobInfo.status.state;
            switch (frameworkJobType) {
                case 'AttemptCreationPending':
                case 'AttemptCreationRequested':
                case 'AttemptPreparing':
                    kubernetesTrialJob.status = 'WAITING';
                    break;
                case 'AttemptRunning':
                    kubernetesTrialJob.status = 'RUNNING';
                    if (kubernetesTrialJob.startTime === undefined) {
                        kubernetesTrialJob.startTime = Date.parse(kubernetesJobInfo.status.startTime);
                    }
                    break;
                case 'Completed': {
                    const completedJobType = kubernetesJobInfo.status.attemptStatus.completionStatus.type.name;
                    switch (completedJobType) {
                        case 'Succeeded':
                            kubernetesTrialJob.status = 'SUCCEEDED';
                            break;
                        case 'Failed':
                            kubernetesTrialJob.status = 'FAILED';
                            break;
                        default:
                    }
                    kubernetesTrialJob.endTime = Date.parse(kubernetesJobInfo.status.completionTime);
                    break;
                }
                default:
            }
        }
        return Promise.resolve();
    }
}
exports.FrameworkControllerJobInfoCollector = FrameworkControllerJobInfoCollector;
