"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.PAIJobInfoCollector = void 0;
const request_1 = __importDefault(require("request"));
const ts_deferred_1 = require("ts-deferred");
const errors_1 = require("common/errors");
const log_1 = require("common/log");
class PAIJobInfoCollector {
    trialJobsMap;
    log = log_1.getLogger('PAIJobInfoCollector');
    statusesNeedToCheck;
    finalStatuses;
    constructor(jobMap) {
        this.trialJobsMap = jobMap;
        this.statusesNeedToCheck = ['RUNNING', 'UNKNOWN', 'WAITING'];
        this.finalStatuses = ['SUCCEEDED', 'FAILED', 'USER_CANCELED', 'SYS_CANCELED', 'EARLY_STOPPED'];
    }
    async retrieveTrialStatus(protocol, token, config) {
        if (config === undefined || token === undefined) {
            return Promise.resolve();
        }
        const updatePaiTrialJobs = [];
        for (const [trialJobId, paiTrialJob] of this.trialJobsMap) {
            if (paiTrialJob === undefined) {
                throw new errors_1.NNIError(errors_1.NNIErrorNames.NOT_FOUND, `trial job id ${trialJobId} not found`);
            }
            updatePaiTrialJobs.push(this.getSinglePAITrialJobInfo(protocol, paiTrialJob, token, config));
        }
        await Promise.all(updatePaiTrialJobs);
    }
    getSinglePAITrialJobInfo(_protocol, paiTrialJob, paiToken, config) {
        const deferred = new ts_deferred_1.Deferred();
        if (!this.statusesNeedToCheck.includes(paiTrialJob.status)) {
            deferred.resolve();
            return deferred.promise;
        }
        const getJobInfoRequest = {
            uri: `${config.host}/rest-server/api/v2/jobs/${config.username}~${paiTrialJob.paiJobName}`,
            method: 'GET',
            json: true,
            headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${paiToken}`
            }
        };
        request_1.default(getJobInfoRequest, (error, response, _body) => {
            if ((error !== undefined && error !== null) || response.statusCode >= 400) {
                if (paiTrialJob.status === 'WAITING' || paiTrialJob.status === 'RUNNING') {
                    paiTrialJob.status = 'UNKNOWN';
                }
            }
            else {
                if (response.body.jobStatus && response.body.jobStatus.state) {
                    switch (response.body.jobStatus.state) {
                        case 'WAITING':
                            paiTrialJob.status = 'WAITING';
                            break;
                        case 'RUNNING':
                            paiTrialJob.status = 'RUNNING';
                            if (paiTrialJob.startTime === undefined) {
                                paiTrialJob.startTime = response.body.jobStatus.appLaunchedTime;
                            }
                            if (paiTrialJob.url === undefined) {
                                if (response.body.jobStatus.appTrackingUrl) {
                                    paiTrialJob.url = response.body.jobStatus.appTrackingUrl;
                                }
                                else {
                                    paiTrialJob.url = paiTrialJob.paiJobDetailUrl;
                                }
                            }
                            break;
                        case 'SUCCEEDED':
                            paiTrialJob.status = 'SUCCEEDED';
                            break;
                        case 'STOPPED':
                        case 'STOPPING':
                            if (paiTrialJob.isEarlyStopped !== undefined) {
                                paiTrialJob.status = paiTrialJob.isEarlyStopped === true ?
                                    'EARLY_STOPPED' : 'USER_CANCELED';
                            }
                            else {
                                paiTrialJob.status = 'SYS_CANCELED';
                            }
                            break;
                        case 'FAILED':
                            paiTrialJob.status = 'FAILED';
                            break;
                        default:
                            paiTrialJob.status = 'UNKNOWN';
                    }
                    if (this.finalStatuses.includes(paiTrialJob.status)) {
                        if (paiTrialJob.startTime === undefined) {
                            paiTrialJob.startTime = response.body.jobStatus.appLaunchedTime;
                        }
                        if (paiTrialJob.endTime === undefined) {
                            paiTrialJob.endTime = response.body.jobStatus.completedTime;
                        }
                        if (paiTrialJob.logPath !== undefined) {
                            if (paiTrialJob.url && paiTrialJob.url !== paiTrialJob.logPath) {
                                paiTrialJob.url += `,${paiTrialJob.logPath}`;
                            }
                            else {
                                paiTrialJob.url = `${paiTrialJob.logPath}`;
                            }
                        }
                    }
                }
            }
            deferred.resolve();
        });
        return deferred.promise;
    }
}
exports.PAIJobInfoCollector = PAIJobInfoCollector;
