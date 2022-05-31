"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.StorageService = void 0;
const fs_1 = __importDefault(require("fs"));
const os_1 = __importDefault(require("os"));
const path_1 = __importDefault(require("path"));
const log_1 = require("common/log");
const utils_1 = require("common/utils");
const util_1 = require("../common/util");
class StorageService {
    localRoot = "";
    remoteRoot = "";
    logger;
    constructor() {
        this.logger = log_1.getLogger('StorageService');
    }
    initialize(localRoot, remoteRoot) {
        this.logger.debug(`Initializing storage to local: ${localRoot} remote: ${remoteRoot}`);
        this.localRoot = localRoot;
        this.remoteRoot = remoteRoot;
    }
    async rename(remotePath, newName) {
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`rename remotePath: ${remotePath} to: ${newName}`);
        await this.internalRename(remotePath, newName);
    }
    async createDirectory(remotePath) {
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`create remotePath: ${remotePath}`);
        await this.internalMkdir(remotePath);
    }
    async copyDirectory(localPath, remotePath, asGzip = false) {
        localPath = this.expandPath(false, localPath);
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`copy localPath: ${localPath} to remotePath: ${remotePath}, asGzip ${asGzip}`);
        if (!await this.internalExists(remotePath)) {
            await this.internalMkdir(remotePath);
        }
        if (asGzip) {
            const localPathBaseName = path_1.default.basename(localPath);
            const tempTarFileName = `nni_tmp_${localPathBaseName}_${utils_1.uniqueString(5)}.tar.gz`;
            const tarFileName = `${localPathBaseName}.tar.gz`;
            const localTarPath = path_1.default.join(os_1.default.tmpdir(), tempTarFileName);
            await util_1.tarAdd(localTarPath, localPath);
            await this.internalCopy(localTarPath, remotePath, false, false, true);
            const remoteFileName = this.internalJoin(remotePath, tempTarFileName);
            await this.internalRename(remoteFileName, tarFileName);
            await fs_1.default.promises.unlink(localTarPath);
            remotePath = this.internalJoin(remotePath, tarFileName);
        }
        else {
            await this.internalCopy(localPath, remotePath, true, false, true);
            remotePath = this.internalJoin(remotePath, path_1.default.basename(localPath));
        }
        return remotePath;
    }
    async copyDirectoryBack(remotePath, localPath) {
        localPath = this.expandPath(false, localPath);
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`copy remotePath: ${remotePath} to localPath: ${localPath}`);
        return await this.internalCopy(remotePath, localPath, true, true, false);
    }
    async removeDirectory(remotePath, isRecursive) {
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`remove remotePath: ${remotePath}`);
        await this.internalRemove(remotePath, true, isRecursive);
    }
    async readFileContent(remotePath, offset = -1, length = -1) {
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`read remote file: ${remotePath}, offset: ${offset}, length: ${length}`);
        return this.internalRead(remotePath, offset, length);
    }
    async listDirectory(remotePath) {
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`list remotePath: ${remotePath}`);
        return await this.internalList(remotePath);
    }
    async exists(remotePath) {
        remotePath = this.expandPath(true, remotePath);
        const exists = await this.internalExists(remotePath);
        this.logger.debug(`exists remotePath: ${remotePath} is ${exists}`);
        return exists;
    }
    async save(content, remotePath, isAttach = false) {
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`saving content to remotePath: ${remotePath}, length: ${content.length}, isAttach: ${isAttach}`);
        const remoteDir = this.internalDirname(remotePath);
        if (isAttach) {
            if (await this.internalExists(remoteDir) === false) {
                await this.internalMkdir(remoteDir);
            }
            const result = await this.internalAttach(remotePath, content);
            if (false === result) {
                throw new Error("this.internalAttach doesn't support");
            }
        }
        else {
            const fileName = this.internalBasename(remotePath);
            const tempFileName = `temp_${utils_1.uniqueString(4)}_${fileName}`;
            const localTempFileName = path_1.default.join(os_1.default.tmpdir(), tempFileName);
            const remoteTempFile = this.internalJoin(remoteDir, tempFileName);
            if (await this.internalExists(remotePath) === true) {
                await this.internalRemove(remotePath, false, false);
            }
            await fs_1.default.promises.writeFile(localTempFileName, content);
            await this.internalCopy(localTempFileName, remoteDir, false, false, true);
            await this.rename(remoteTempFile, fileName);
            await fs_1.default.promises.unlink(localTempFileName);
        }
    }
    async copyFile(localPath, remotePath) {
        localPath = this.expandPath(false, localPath);
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`copying file localPath: ${localPath} to remotePath: ${remotePath}`);
        await this.internalCopy(localPath, remotePath, false, false, true);
    }
    async copyFileBack(remotePath, localPath) {
        localPath = this.expandPath(false, localPath);
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`copy file remotePath: ${remotePath} to localPath: ${localPath}`);
        await this.internalCopy(remotePath, localPath, false, true, false);
    }
    async removeFile(remotePath) {
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`remove file remotePath: ${remotePath}`);
        await this.internalRemove(remotePath, false, false);
    }
    joinPath(...paths) {
        let fullPath = this.internalJoin(...paths);
        if (this.internalIsRelativePath(fullPath) === true && this.remoteRoot !== "") {
            fullPath = this.internalJoin(this.remoteRoot, fullPath);
        }
        return fullPath;
    }
    expandPath(isRemote, ...paths) {
        let normalizedPath;
        if (isRemote) {
            normalizedPath = this.joinPath(...paths);
        }
        else {
            normalizedPath = path_1.default.join(...paths);
            if (!path_1.default.isAbsolute(normalizedPath) && this.localRoot !== "") {
                normalizedPath = path_1.default.join(this.localRoot, normalizedPath);
            }
        }
        return normalizedPath;
    }
}
exports.StorageService = StorageService;
