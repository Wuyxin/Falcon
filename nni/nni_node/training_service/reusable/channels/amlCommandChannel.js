"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.AMLCommandChannel = void 0;
const utils_1 = require("common/utils");
const commandChannel_1 = require("../commandChannel");
class AMLRunnerConnection extends commandChannel_1.RunnerConnection {
}
class AMLCommandChannel extends commandChannel_1.CommandChannel {
    stopping = false;
    sendQueues = [];
    get channelName() {
        return "aml";
    }
    async config(_key, _value) {
    }
    async start() {
    }
    async stop() {
        this.stopping = true;
    }
    async run() {
        await Promise.all([
            this.receiveLoop(),
            this.sendLoop()
        ]);
    }
    async sendCommandInternal(environment, message) {
        this.sendQueues.push([environment, message]);
    }
    createRunnerConnection(environment) {
        return new AMLRunnerConnection(environment);
    }
    async sendLoop() {
        const intervalSeconds = 0.5;
        while (!this.stopping) {
            const start = new Date();
            if (this.sendQueues.length > 0) {
                while (this.sendQueues.length > 0) {
                    const item = this.sendQueues.shift();
                    if (item === undefined) {
                        break;
                    }
                    const environment = item[0];
                    const message = item[1];
                    const amlClient = environment.amlClient;
                    if (!amlClient) {
                        throw new Error('aml client not initialized!');
                    }
                    amlClient.sendCommand(message);
                }
            }
            const end = new Date();
            const delayMs = intervalSeconds * 1000 - (end.valueOf() - start.valueOf());
            if (delayMs > 0) {
                await utils_1.delay(delayMs);
            }
        }
    }
    async receiveLoop() {
        const intervalSeconds = 2;
        while (!this.stopping) {
            const start = new Date();
            const runnerConnections = [...this.runnerConnections.values()];
            for (const runnerConnection of runnerConnections) {
                const amlEnvironmentInformation = runnerConnection.environment;
                const amlClient = amlEnvironmentInformation.amlClient;
                let currentMessageIndex = amlEnvironmentInformation.currentMessageIndex;
                if (!amlClient) {
                    throw new Error('AML client not initialized!');
                }
                const command = await amlClient.receiveCommand();
                if (command && Object.prototype.hasOwnProperty.call(command, "trial_runner")) {
                    const messages = command['trial_runner'];
                    if (messages) {
                        if (messages instanceof Object && currentMessageIndex < messages.length - 1) {
                            for (let index = currentMessageIndex + 1; index < messages.length; index++) {
                                this.handleCommand(runnerConnection.environment, messages[index]);
                            }
                            currentMessageIndex = messages.length - 1;
                        }
                        else if (currentMessageIndex === -1) {
                            this.handleCommand(runnerConnection.environment, messages);
                            currentMessageIndex += 1;
                        }
                        amlEnvironmentInformation.currentMessageIndex = currentMessageIndex;
                    }
                }
            }
            const end = new Date();
            const delayMs = intervalSeconds * 1000 - (end.valueOf() - start.valueOf());
            if (delayMs > 0) {
                await utils_1.delay(delayMs);
            }
        }
    }
}
exports.AMLCommandChannel = AMLCommandChannel;
