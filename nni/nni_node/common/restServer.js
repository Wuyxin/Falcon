"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.LegacyRestServer = void 0;
const assert_1 = __importDefault(require("assert"));
const express_1 = __importDefault(require("express"));
const ts_deferred_1 = require("ts-deferred");
const log_1 = require("./log");
const experimentStartupInfo_1 = require("./experimentStartupInfo");
class LegacyRestServer {
    startTask;
    stopTask;
    server;
    hostName = '0.0.0.0';
    port;
    app = express_1.default();
    log = log_1.getLogger('RestServer');
    basePort;
    constructor() {
        this.port = experimentStartupInfo_1.getBasePort();
        assert_1.default(this.port && this.port > 1024);
    }
    get endPoint() {
        return `http://${this.hostName}:${this.port}`;
    }
    start(hostName) {
        this.log.info(`RestServer start`);
        if (this.startTask !== undefined) {
            return this.startTask.promise;
        }
        this.startTask = new ts_deferred_1.Deferred();
        this.registerRestHandler();
        if (hostName) {
            this.hostName = hostName;
        }
        this.log.info(`RestServer base port is ${this.port}`);
        this.server = this.app.listen(this.port, this.hostName).on('listening', () => {
            this.startTask.resolve();
        }).on('error', (e) => {
            this.startTask.reject(e);
        });
        return this.startTask.promise;
    }
    stop() {
        if (this.stopTask !== undefined) {
            return this.stopTask.promise;
        }
        this.stopTask = new ts_deferred_1.Deferred();
        if (this.startTask === undefined) {
            this.stopTask.resolve();
            return this.stopTask.promise;
        }
        else {
            this.startTask.promise.then(() => {
                this.server.close().on('close', () => {
                    this.log.info('Rest server stopped.');
                    this.stopTask.resolve();
                }).on('error', (e) => {
                    this.log.error(`Error occurred stopping Rest server: ${e.message}`);
                    this.stopTask.reject();
                });
            }, () => {
                this.stopTask.resolve();
            });
        }
        this.stopTask.resolve();
        return this.stopTask.promise;
    }
}
exports.LegacyRestServer = LegacyRestServer;
