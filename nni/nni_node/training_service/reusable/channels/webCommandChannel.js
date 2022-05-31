"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.WebCommandChannel = void 0;
const ws_1 = require("ws");
const experimentStartupInfo_1 = require("common/experimentStartupInfo");
const commands_1 = require("core/commands");
const commandChannel_1 = require("../commandChannel");
class WebRunnerConnection extends commandChannel_1.RunnerConnection {
    clients = [];
    async close() {
        await super.close();
        while (this.clients.length > 0) {
            const client = this.clients.shift();
            if (client !== undefined) {
                client.close();
            }
        }
    }
    AddClient(client) {
        this.clients.push(client);
    }
}
class WebCommandChannel extends commandChannel_1.CommandChannel {
    expId = experimentStartupInfo_1.getExperimentId();
    static commandChannel;
    webSocketServer;
    clients = new Map();
    get channelName() {
        return "web";
    }
    async config(_key, _value) {
    }
    constructor(commandEmitter) {
        super(commandEmitter);
    }
    static getInstance(commandEmitter) {
        if (!this.commandChannel) {
            this.commandChannel = new WebCommandChannel(commandEmitter);
        }
        return this.commandChannel;
    }
    async start() {
        const port = experimentStartupInfo_1.getBasePort() + 1;
        this.webSocketServer = new ws_1.Server({ port });
        this.webSocketServer.on('connection', (client) => {
            this.log.debug(`WebCommandChannel: received connection`);
            client.onerror = (event) => {
                this.log.error('error on client', event);
            };
            this.clients.set(client, undefined);
            client.onmessage = (message) => {
                this.receivedWebSocketMessage(client, message);
            };
        }).on('error', (error) => {
            this.log.error(`error on websocket server ${error}`);
        });
    }
    async stop() {
        if (this.webSocketServer !== undefined) {
            this.webSocketServer.close();
        }
    }
    async run() {
    }
    async sendCommandInternal(environment, message) {
        if (this.webSocketServer === undefined) {
            throw new Error(`WebCommandChannel: uninitialized!`);
        }
        const runnerConnection = this.runnerConnections.get(environment.id);
        if (runnerConnection !== undefined) {
            for (const client of runnerConnection.clients) {
                client.send(message);
            }
        }
        else {
            this.log.warning(`WebCommandChannel: cannot find client for env ${environment.id}, message is ignored.`);
        }
    }
    createRunnerConnection(environment) {
        return new WebRunnerConnection(environment);
    }
    receivedWebSocketMessage(client, message) {
        let connection = this.clients.get(client);
        const rawCommands = message.data.toString();
        if (connection === undefined) {
            const commands = this.parseCommands(rawCommands);
            let isValid = false;
            this.log.debug('WebCommandChannel: received initialize message:', rawCommands);
            if (commands.length > 0) {
                const commandType = commands[0][0];
                const result = commands[0][1];
                if (commandType === commands_1.INITIALIZED &&
                    result.expId === this.expId &&
                    this.runnerConnections.has(result.runnerId)) {
                    const runnerConnection = this.runnerConnections.get(result.runnerId);
                    this.clients.set(client, runnerConnection);
                    runnerConnection.AddClient(client);
                    connection = runnerConnection;
                    isValid = true;
                    this.log.debug(`WebCommandChannel: client of env ${runnerConnection.environment.id} initialized`);
                }
                else {
                    this.log.warning(`WebCommandChannel: client is not initialized, runnerId: ${result.runnerId}, command: ${commandType}, expId: ${this.expId}, exists: ${this.runnerConnections.has(result.runnerId)}`);
                }
            }
            if (!isValid) {
                this.log.warning(`WebCommandChannel: rejected client with invalid init message ${rawCommands}`);
                client.close();
                this.clients.delete(client);
            }
        }
        if (connection !== undefined) {
            this.handleCommand(connection.environment, rawCommands);
        }
    }
}
exports.WebCommandChannel = WebCommandChannel;
