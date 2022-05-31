"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    Object.defineProperty(o, k2, { enumerable: true, get: function() { return m[k]; } });
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.FileCommandChannel = void 0;
const component = __importStar(require("common/component"));
const utils_1 = require("common/utils");
const commandChannel_1 = require("../commandChannel");
const storageService_1 = require("../storageService");
class FileHandler {
    fileName;
    offset = 0;
    constructor(fileName) {
        this.fileName = fileName;
    }
}
class FileRunnerConnection extends commandChannel_1.RunnerConnection {
    handlers = new Map();
}
class FileCommandChannel extends commandChannel_1.CommandChannel {
    commandPath = "commands";
    stopping = false;
    sendQueues = [];
    get channelName() {
        return "file";
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
        return new FileRunnerConnection(environment);
    }
    async sendLoop() {
        const intervalSeconds = 0.5;
        while (!this.stopping) {
            const start = new Date();
            if (this.sendQueues.length > 0) {
                const storageService = component.get(storageService_1.StorageService);
                while (this.sendQueues.length > 0) {
                    const item = this.sendQueues.shift();
                    if (item === undefined) {
                        break;
                    }
                    const environment = item[0];
                    const message = `${item[1]}\n`;
                    const fileName = storageService.joinPath(environment.workingFolder, this.commandPath, `manager_commands.txt`);
                    await storageService.save(message, fileName, true);
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
        const storageService = component.get(storageService_1.StorageService);
        while (!this.stopping) {
            const start = new Date();
            const runnerConnections = [...this.runnerConnections.values()];
            for (const runnerConnection of runnerConnections) {
                const envCommandFolder = storageService.joinPath(runnerConnection.environment.workingFolder, this.commandPath);
                if (runnerConnection.handlers.size < runnerConnection.environment.nodeCount) {
                    const commandFileNames = await storageService.listDirectory(envCommandFolder);
                    const toAddedFileNames = [];
                    for (const commandFileName of commandFileNames) {
                        if (commandFileName.startsWith("runner_commands") && !runnerConnection.handlers.has(commandFileName)) {
                            toAddedFileNames.push(commandFileName);
                        }
                    }
                    for (const toAddedFileName of toAddedFileNames) {
                        const fullPath = storageService.joinPath(envCommandFolder, toAddedFileName);
                        const fileHandler = new FileHandler(fullPath);
                        runnerConnection.handlers.set(toAddedFileName, fileHandler);
                        this.log.debug(`FileCommandChannel: added fileHandler env ${runnerConnection.environment.id} ${toAddedFileName}`);
                    }
                }
                for (const fileHandler of runnerConnection.handlers.values()) {
                    const newContent = await storageService.readFileContent(fileHandler.fileName, fileHandler.offset, undefined);
                    if (newContent.length > 0) {
                        const commands = newContent.split('\n');
                        for (const command of commands) {
                            this.handleCommand(runnerConnection.environment, command);
                        }
                        fileHandler.offset += newContent.length;
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
exports.FileCommandChannel = FileCommandChannel;
