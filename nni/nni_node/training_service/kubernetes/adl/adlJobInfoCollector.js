"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.AdlJobInfoCollector = void 0;
const kubernetesJobInfoCollector_1 = require("../kubernetesJobInfoCollector");
class AdlJobInfoCollector extends kubernetesJobInfoCollector_1.KubernetesJobInfoCollector {
    constructor(jobMap) {
        super(jobMap);
    }
    async retrieveSingleTrialJobInfo(adlClient, kubernetesTrialJob) {
        if (!this.statusesNeedToCheck.includes(kubernetesTrialJob.status)) {
            return Promise.resolve();
        }
        if (adlClient === undefined) {
            return Promise.reject('AdlClient is undefined');
        }
        let kubernetesJobInfo;
        let kubernetesPodsInfo;
        try {
            kubernetesJobInfo = await adlClient.getKubernetesJob(kubernetesTrialJob.kubernetesJobName);
            kubernetesPodsInfo = await adlClient.getKubernetesPods(kubernetesTrialJob.kubernetesJobName);
        }
        catch (error) {
            this.log.error(`Get job ${kubernetesTrialJob.kubernetesJobName} info failed, error is ${error}`);
            return Promise.resolve();
        }
        if (kubernetesJobInfo.status) {
            const phase = kubernetesJobInfo.status.phase;
            switch (phase) {
                case 'Pending':
                case 'Starting':
                    kubernetesTrialJob.status = 'WAITING';
                    if (kubernetesPodsInfo.items.length > 0) {
                        if (kubernetesPodsInfo.items[0].status.containerStatuses != undefined) {
                            const currState = kubernetesPodsInfo.items[0].status.containerStatuses[0].state;
                            if (currState.waiting != undefined) {
                                const msg = currState.waiting.reason;
                                if (msg == "ImagePullBackOff" || msg == "ErrImagePull") {
                                    kubernetesTrialJob.status = 'FAILED';
                                }
                            }
                        }
                        kubernetesTrialJob.message = kubernetesPodsInfo.items
                            .map((pod) => JSON.stringify(pod.status.containerStatuses))
                            .join('\n');
                    }
                    kubernetesTrialJob.startTime = Date.parse(kubernetesJobInfo.metadata.creationTimestamp);
                    break;
                case 'Running':
                case 'Stopping':
                    kubernetesTrialJob.status = 'RUNNING';
                    kubernetesTrialJob.message = `Use 'nnictl log trial --trial_id ${kubernetesTrialJob.id}' to check the log stream.`;
                    if (kubernetesTrialJob.startTime === undefined) {
                        kubernetesTrialJob.startTime = Date.parse(kubernetesJobInfo.metadata.creationTimestamp);
                    }
                    break;
                case 'Failed':
                    kubernetesTrialJob.status = 'FAILED';
                    kubernetesTrialJob.message = kubernetesJobInfo.status.message;
                    if (kubernetesPodsInfo.items.length > 0) {
                        kubernetesTrialJob.message += " ; ";
                        kubernetesTrialJob.message += `Use 'nnictl log trial --trial_id ${kubernetesTrialJob.id}' for the path of the collected logs.`;
                    }
                    kubernetesTrialJob.endTime = Date.parse(kubernetesJobInfo.status.completionTimestamp);
                    break;
                case 'Succeeded':
                    kubernetesTrialJob.status = 'SUCCEEDED';
                    kubernetesTrialJob.endTime = Date.parse(kubernetesJobInfo.status.completionTimestamp);
                    kubernetesTrialJob.message = `Succeeded at ${kubernetesJobInfo.status.completionTimestamp}`;
                    break;
                default:
            }
        }
        return Promise.resolve();
    }
}
exports.AdlJobInfoCollector = AdlJobInfoCollector;
