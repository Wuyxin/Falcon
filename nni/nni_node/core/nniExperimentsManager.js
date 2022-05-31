"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.NNIExperimentsManager = void 0;
const fs_1 = __importDefault(require("fs"));
const os_1 = __importDefault(require("os"));
const path_1 = __importDefault(require("path"));
const assert_1 = __importDefault(require("assert"));
const log_1 = require("../common/log");
const utils_1 = require("../common/utils");
const ts_deferred_1 = require("ts-deferred");
class NNIExperimentsManager {
    experimentsPath;
    log;
    profileUpdateTimer;
    constructor() {
        this.experimentsPath = utils_1.getExperimentsInfoPath();
        this.log = log_1.getLogger('NNIExperimentsManager');
        this.profileUpdateTimer = {};
    }
    async getExperimentsInfo() {
        const fileInfo = await this.withLockIterated(this.readExperimentsInfo, 100);
        const experimentsInformation = JSON.parse(fileInfo.buffer.toString());
        const expIdList = Object.keys(experimentsInformation).filter((expId) => {
            return experimentsInformation[expId]['status'] !== 'STOPPED';
        });
        const updateList = (await Promise.all(expIdList.map((expId) => {
            return this.checkCrashed(expId, experimentsInformation[expId]['pid']);
        }))).filter(crashedInfo => crashedInfo.isCrashed);
        if (updateList.length > 0) {
            const result = await this.withLockIterated(this.updateAllStatus, 100, updateList.map(crashedInfo => crashedInfo.experimentId), fileInfo.mtime);
            if (result !== undefined) {
                return JSON.parse(JSON.stringify(Object.keys(result).map(key => result[key])));
            }
            else {
                await utils_1.delay(500);
                return await this.getExperimentsInfo();
            }
        }
        else {
            return JSON.parse(JSON.stringify(Object.keys(experimentsInformation).map(key => experimentsInformation[key])));
        }
    }
    setExperimentPath(newPath) {
        if (newPath[0] === '~') {
            newPath = path_1.default.join(os_1.default.homedir(), newPath.slice(1));
        }
        if (!path_1.default.isAbsolute(newPath)) {
            newPath = path_1.default.resolve(newPath);
        }
        this.log.info(`Set new experiment information path: ${newPath}`);
        this.experimentsPath = newPath;
    }
    setExperimentInfo(experimentId, key, value) {
        try {
            if (this.profileUpdateTimer[key] !== undefined) {
                clearTimeout(this.profileUpdateTimer[key]);
                this.profileUpdateTimer[key] = undefined;
            }
            this.withLockSync(() => {
                const experimentsInformation = JSON.parse(fs_1.default.readFileSync(this.experimentsPath).toString());
                assert_1.default(experimentId in experimentsInformation, `Experiment Manager: Experiment Id ${experimentId} not found, this should not happen`);
                if (value !== undefined) {
                    experimentsInformation[experimentId][key] = value;
                }
                else {
                    delete experimentsInformation[experimentId][key];
                }
                fs_1.default.writeFileSync(this.experimentsPath, JSON.stringify(experimentsInformation, null, 4));
            });
        }
        catch (err) {
            this.log.error(err);
            this.log.debug(`Experiment Manager: Retry set key value: ${experimentId} {${key}: ${value}}`);
            if (err.code === 'EEXIST' || err.message === 'File has been locked.') {
                this.profileUpdateTimer[key] = setTimeout(this.setExperimentInfo.bind(this), 100, experimentId, key, value);
            }
        }
    }
    async withLockIterated(func, retry, ...args) {
        if (retry < 0) {
            throw new Error('Lock file out of retries.');
        }
        try {
            return this.withLockSync(func, ...args);
        }
        catch (err) {
            if (err.code === 'EEXIST' || err.message === 'File has been locked.') {
                await utils_1.delay(50);
                return await this.withLockIterated(func, retry - 1, ...args);
            }
            throw err;
        }
    }
    withLockSync(func, ...args) {
        return utils_1.withLockSync(func.bind(this), this.experimentsPath, { stale: 2 * 1000 }, ...args);
    }
    readExperimentsInfo() {
        const buffer = fs_1.default.readFileSync(this.experimentsPath);
        const mtime = fs_1.default.statSync(this.experimentsPath).mtimeMs;
        return { buffer: buffer, mtime: mtime };
    }
    async checkCrashed(expId, pid) {
        const alive = await utils_1.isAlive(pid);
        return { experimentId: expId, isCrashed: !alive };
    }
    updateAllStatus(updateList, timestamp) {
        if (timestamp !== fs_1.default.statSync(this.experimentsPath).mtimeMs) {
            return;
        }
        else {
            const experimentsInformation = JSON.parse(fs_1.default.readFileSync(this.experimentsPath).toString());
            updateList.forEach((expId) => {
                if (experimentsInformation[expId]) {
                    experimentsInformation[expId]['status'] = 'STOPPED';
                    delete experimentsInformation[expId]['port'];
                }
                else {
                    this.log.error(`Experiment Manager: Experiment Id ${expId} not found, this should not happen`);
                }
            });
            fs_1.default.writeFileSync(this.experimentsPath, JSON.stringify(experimentsInformation, null, 4));
            return experimentsInformation;
        }
    }
    async stop() {
        this.log.debug('Stopping experiment manager.');
        await this.cleanUp().catch(err => this.log.error(err.message));
        this.log.debug('Experiment manager stopped.');
    }
    async cleanUp() {
        const deferred = new ts_deferred_1.Deferred();
        if (this.isUndone()) {
            this.log.debug('Experiment manager: something undone');
            setTimeout(((deferred) => {
                if (this.isUndone()) {
                    deferred.reject(new Error('Still has undone after 5s, forced stop.'));
                }
                else {
                    deferred.resolve();
                }
            }).bind(this), 5 * 1000, deferred);
        }
        else {
            this.log.debug('Experiment manager: all clean up');
            deferred.resolve();
        }
        return deferred.promise;
    }
    isUndone() {
        return Object.keys(this.profileUpdateTimer).filter((key) => {
            return this.profileUpdateTimer[key] !== undefined;
        }).length > 0;
    }
}
exports.NNIExperimentsManager = NNIExperimentsManager;
