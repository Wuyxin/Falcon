"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.AMLClient = void 0;
const ts_deferred_1 = require("ts-deferred");
const python_shell_1 = require("python-shell");
class AMLClient {
    subscriptionId;
    resourceGroup;
    workspaceName;
    experimentId;
    image;
    scriptName;
    pythonShellClient;
    codeDir;
    computeTarget;
    constructor(subscriptionId, resourceGroup, workspaceName, experimentId, computeTarget, image, scriptName, codeDir) {
        this.subscriptionId = subscriptionId;
        this.resourceGroup = resourceGroup;
        this.workspaceName = workspaceName;
        this.experimentId = experimentId;
        this.image = image;
        this.scriptName = scriptName;
        this.codeDir = codeDir;
        this.computeTarget = computeTarget;
    }
    submit() {
        const deferred = new ts_deferred_1.Deferred();
        this.pythonShellClient = new python_shell_1.PythonShell('amlUtil.py', {
            scriptPath: './config/aml',
            pythonPath: process.platform === 'win32' ? 'python' : 'python3',
            pythonOptions: ['-u'],
            args: [
                '--subscription_id', this.subscriptionId,
                '--resource_group', this.resourceGroup,
                '--workspace_name', this.workspaceName,
                '--compute_target', this.computeTarget,
                '--docker_image', this.image,
                '--experiment_name', `nni_exp_${this.experimentId}`,
                '--script_dir', this.codeDir,
                '--script_name', this.scriptName
            ]
        });
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
        const deferred = new ts_deferred_1.Deferred();
        this.pythonShellClient.send('stop');
        this.pythonShellClient.on('message', (result) => {
            const stopResult = this.parseContent('stop_result', result);
            if (stopResult === 'success') {
                deferred.resolve(true);
            }
            else if (stopResult === 'failed') {
                deferred.resolve(false);
            }
        });
        return deferred.promise;
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
exports.AMLClient = AMLClient;
