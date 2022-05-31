"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.DlcClient = void 0;
const ts_deferred_1 = require("ts-deferred");
const python_shell_1 = require("python-shell");
const log_1 = require("common/log");
class DlcClient {
    log;
    type;
    image;
    jobType;
    podCount;
    ecsSpec;
    region;
    nasDataSourceId;
    ossDataSourceId;
    accessKeyId;
    accessKeySecret;
    experimentId;
    environmentId;
    userCommand;
    logDir;
    pythonShellClient;
    constructor(type, image, jobType, podCount, experimentId, environmentId, ecsSpec, region, nasDataSourceId, accessKeyId, accessKeySecret, userCommand, logDir, ossDataSourceId) {
        this.log = log_1.getLogger('DlcClient');
        this.type = type;
        this.image = image;
        this.jobType = jobType;
        this.podCount = podCount;
        this.ecsSpec = ecsSpec;
        this.image = image;
        this.region = region;
        this.nasDataSourceId = nasDataSourceId;
        if (ossDataSourceId !== undefined) {
            this.ossDataSourceId = ossDataSourceId;
        }
        else {
            this.ossDataSourceId = '';
        }
        this.accessKeyId = accessKeyId;
        this.accessKeySecret = accessKeySecret;
        this.experimentId = experimentId;
        this.environmentId = environmentId;
        this.userCommand = userCommand;
        this.logDir = logDir;
    }
    submit() {
        const deferred = new ts_deferred_1.Deferred();
        this.pythonShellClient = new python_shell_1.PythonShell('dlcUtil.py', {
            scriptPath: './config/dlc',
            pythonPath: 'python3',
            pythonOptions: ['-u'],
            args: [
                '--type', this.type,
                '--image', this.image,
                '--job_type', this.jobType,
                '--pod_count', String(this.podCount),
                '--ecs_spec', this.ecsSpec,
                '--region', this.region,
                '--nas_data_source_id', this.nasDataSourceId,
                '--oss_data_source_id', this.ossDataSourceId,
                '--access_key_id', this.accessKeyId,
                '--access_key_secret', this.accessKeySecret,
                '--experiment_name', `nni_exp_${this.experimentId}_env_${this.environmentId}`,
                '--user_command', this.userCommand,
                '--log_dir', this.logDir,
            ]
        });
        this.log.debug(this.pythonShellClient.command);
        this.pythonShellClient.on('message', function (envId) {
            deferred.resolve(envId);
        });
        this.monitorError(this.pythonShellClient, deferred);
        return deferred.promise;
    }
    stop() {
        if (this.pythonShellClient === undefined) {
            throw Error('python shell client not initialized!');
        }
        this.pythonShellClient.send('stop');
    }
    getTrackingUrl() {
        const deferred = new ts_deferred_1.Deferred();
        if (this.pythonShellClient === undefined) {
            throw Error('python shell client not initialized!');
        }
        this.pythonShellClient.send('tracking_url');
        this.pythonShellClient.on('message', (status) => {
            const trackingUrl = this.parseContent('tracking_url', status);
            if (trackingUrl !== '') {
                deferred.resolve(trackingUrl);
            }
        });
        this.monitorError(this.pythonShellClient, deferred);
        return deferred.promise;
    }
    updateStatus(oldStatus) {
        const deferred = new ts_deferred_1.Deferred();
        if (this.pythonShellClient === undefined) {
            throw Error('python shell client not initialized!');
        }
        this.pythonShellClient.send('update_status');
        this.pythonShellClient.on('message', (status) => {
            let newStatus = this.parseContent('status', status);
            if (newStatus === '') {
                newStatus = oldStatus;
            }
            deferred.resolve(newStatus);
        });
        this.monitorError(this.pythonShellClient, deferred);
        return deferred.promise;
    }
    sendCommand(message) {
        if (this.pythonShellClient === undefined) {
            throw Error('python shell client not initialized!');
        }
        this.log.debug(`command:${message}`);
        this.pythonShellClient.send(`command:${message}`);
    }
    receiveCommand() {
        const deferred = new ts_deferred_1.Deferred();
        if (this.pythonShellClient === undefined) {
            throw Error('python shell client not initialized!');
        }
        this.pythonShellClient.send('receive');
        this.pythonShellClient.on('message', (command) => {
            const message = this.parseContent('receive', command);
            if (message !== '') {
                deferred.resolve(JSON.parse(message));
            }
        });
        this.monitorError(this.pythonShellClient, deferred);
        return deferred.promise;
    }
    monitorError(pythonShellClient, deferred) {
        pythonShellClient.on('error', function (error) {
            deferred.reject(error);
        });
        pythonShellClient.on('close', function (error) {
            deferred.reject(error);
        });
    }
    parseContent(head, command) {
        const items = command.split(':');
        if (items[0] === head) {
            return command.slice(head.length + 1);
        }
        return '';
    }
}
exports.DlcClient = DlcClient;
