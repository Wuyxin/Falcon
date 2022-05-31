"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.CommandChannel = exports.RunnerConnection = exports.Command = void 0;
const log_1 = require("common/log");
const commands_1 = require("core/commands");
const ipcInterface_1 = require("core/ipcInterface");
const acceptedCommands = new Set(commands_1.TRIAL_COMMANDS);
class Command {
    environment;
    command;
    data;
    constructor(environment, command, data) {
        if (!acceptedCommands.has(command)) {
            throw new Error(`unaccepted command ${command}`);
        }
        this.environment = environment;
        this.command = command;
        this.data = data;
    }
}
exports.Command = Command;
class RunnerConnection {
    environment;
    constructor(environment) {
        this.environment = environment;
    }
    async open() {
    }
    async close() {
    }
}
exports.RunnerConnection = RunnerConnection;
class CommandChannel {
    log;
    runnerConnections = new Map();
    commandEmitter;
    commandPattern = /(?<type>[\w]{2})(?<length>[\d]{14})(?<data>.*)\n?/gm;
    constructor(commandEmitter) {
        this.log = log_1.getLogger('CommandChannel');
        this.commandEmitter = commandEmitter;
    }
    async sendCommand(environment, commandType, data) {
        const command = ipcInterface_1.encodeCommand(commandType, JSON.stringify(data));
        this.log.debug(`CommandChannel: env ${environment.id} sending command: ${command}`);
        await this.sendCommandInternal(environment, command.toString("utf8"));
    }
    async open(environment) {
        if (this.runnerConnections.has(environment.id)) {
            throw new Error(`CommandChannel: env ${environment.id} is opened already, shouldn't be opened again.`);
        }
        const connection = this.createRunnerConnection(environment);
        this.runnerConnections.set(environment.id, connection);
        await connection.open();
    }
    async close(environment) {
        if (this.runnerConnections.has(environment.id)) {
            const connection = this.runnerConnections.get(environment.id);
            this.runnerConnections.delete(environment.id);
            if (connection !== undefined) {
                await connection.close();
            }
        }
    }
    parseCommands(content) {
        const commands = [];
        let matches = this.commandPattern.exec(content);
        while (matches) {
            if (undefined !== matches.groups) {
                const commandType = matches.groups["type"];
                const dataLength = parseInt(matches.groups["length"]);
                const data = matches.groups["data"];
                if (dataLength !== data.length) {
                    throw new Error(`dataLength ${dataLength} not equal to actual length ${data.length}: ${data}`);
                }
                try {
                    const finalData = JSON.parse(data);
                    commands.push([commandType, finalData]);
                }
                catch (error) {
                    this.log.error(`CommandChannel: error on parseCommands ${error}, original: ${matches.groups["data"]}`);
                    throw error;
                }
            }
            matches = this.commandPattern.exec(content);
        }
        return commands;
    }
    handleCommand(environment, content) {
        const parsedResults = this.parseCommands(content);
        for (const parsedResult of parsedResults) {
            const commandType = parsedResult[0];
            const data = parsedResult[1];
            const command = new Command(environment, commandType, data);
            this.commandEmitter.emit("command", command);
            this.log.trace(`CommandChannel: env ${environment.id} emit command: ${commandType}, ${data}.`);
        }
    }
}
exports.CommandChannel = CommandChannel;
