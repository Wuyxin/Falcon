"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.MountedStorageService = void 0;
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
const ts_deferred_1 = require("ts-deferred");
const storageService_1 = require("../storageService");
class MountedStorageService extends storageService_1.StorageService {
    internalConfig(_key, _value) {
    }
    async internalRemove(path, isDirectory, isRecursive) {
        if (isDirectory) {
            if (isRecursive) {
                const children = await fs_1.default.promises.readdir(path);
                for (const file of children) {
                    const filePath = this.internalJoin(path, file);
                    const stat = await fs_1.default.promises.lstat(filePath);
                    await this.internalRemove(filePath, stat.isDirectory(), isRecursive);
                }
            }
            await fs_1.default.promises.rmdir(path);
        }
        else {
            await fs_1.default.promises.unlink(path);
        }
    }
    async internalRename(remotePath, newName) {
        const dirName = path_1.default.dirname(remotePath);
        newName = this.internalJoin(dirName, newName);
        await fs_1.default.promises.rename(remotePath, newName);
    }
    async internalMkdir(remotePath) {
        if (!fs_1.default.existsSync(remotePath)) {
            await fs_1.default.promises.mkdir(remotePath, { recursive: true });
        }
    }
    async internalCopy(sourcePath, targetPath, isDirectory, isFromRemote = false, isToRemote = true) {
        if (sourcePath === targetPath) {
            return targetPath;
        }
        this.logger.debug(`copying ${sourcePath} to ${targetPath}, dir ${isDirectory}, isFromRemote ${isFromRemote}, isToRemote: ${isToRemote}`);
        if (isDirectory) {
            const basename = isFromRemote ? this.internalBasename(sourcePath) : path_1.default.basename(sourcePath);
            if (isToRemote) {
                targetPath = this.internalJoin(targetPath, basename);
                await this.internalMkdir(targetPath);
            }
            else {
                targetPath = path_1.default.join(targetPath, basename);
                await fs_1.default.promises.mkdir(targetPath);
            }
            const children = await fs_1.default.promises.readdir(sourcePath);
            for (const child of children) {
                const childSourcePath = this.internalJoin(sourcePath, child);
                const stat = await fs_1.default.promises.lstat(childSourcePath);
                await this.internalCopy(childSourcePath, targetPath, stat.isDirectory(), isFromRemote, isToRemote);
            }
            return targetPath;
        }
        else {
            await this.internalMkdir(targetPath);
            const targetFileName = path_1.default.join(targetPath, path_1.default.basename(sourcePath));
            await fs_1.default.promises.copyFile(sourcePath, targetFileName);
            return targetFileName;
        }
    }
    async internalExists(remotePath) {
        const deferred = new ts_deferred_1.Deferred();
        fs_1.default.exists(remotePath, (exists) => {
            deferred.resolve(exists);
        });
        return deferred.promise;
    }
    async internalRead(remotePath, offset, length) {
        const deferred = new ts_deferred_1.Deferred();
        const maxLength = 1024 * 1024;
        if (offset === undefined) {
            offset = -1;
        }
        const current = offset < 0 ? 0 : offset;
        if (length === undefined) {
            length = -1;
        }
        const readLength = length < 0 ? maxLength : length;
        let result = "";
        const stream = fs_1.default.createReadStream(remotePath, {
            encoding: "utf8",
            start: current,
            end: readLength + current - 1,
        }).on("data", (data) => {
            result += data;
        }).on("end", () => {
            stream.close();
            deferred.resolve(result);
        }).on("error", (err) => {
            deferred.reject(err);
        });
        return deferred.promise;
    }
    async internalList(remotePath) {
        let results = [];
        if (await this.internalExists(remotePath) === true) {
            results = await fs_1.default.promises.readdir(remotePath);
        }
        return results;
    }
    async internalAttach(remotePath, content) {
        await fs_1.default.promises.appendFile(remotePath, content, {
            encoding: "utf8",
            flag: "a",
        });
        return true;
    }
    internalIsRelativePath(remotePath) {
        return !path_1.default.isAbsolute(remotePath);
    }
    internalJoin(...paths) {
        return path_1.default.join(...paths);
    }
    internalDirname(remotePath) {
        return path_1.default.dirname(remotePath);
    }
    internalBasename(remotePath) {
        return path_1.default.basename(remotePath);
    }
}
exports.MountedStorageService = MountedStorageService;
