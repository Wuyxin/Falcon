"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.AzureStorageClientUtility = void 0;
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
const ts_deferred_1 = require("ts-deferred");
const typescript_string_operations_1 = require("typescript-string-operations");
const log_1 = require("common/log");
const utils_1 = require("common/utils");
var AzureStorageClientUtility;
(function (AzureStorageClientUtility) {
    async function createShare(fileServerClient, azureShare) {
        const deferred = new ts_deferred_1.Deferred();
        fileServerClient.createShareIfNotExists(azureShare, (error, _result, _response) => {
            if (error) {
                log_1.getLogger('AzureStorageClientUtility')
                    .error(`Create share failed:, ${error}`);
                deferred.resolve(false);
            }
            else {
                deferred.resolve(true);
            }
        });
        return deferred.promise;
    }
    AzureStorageClientUtility.createShare = createShare;
    async function createDirectory(fileServerClient, azureFoler, azureShare) {
        const deferred = new ts_deferred_1.Deferred();
        fileServerClient.createDirectoryIfNotExists(azureShare, azureFoler, (error, _result, _response) => {
            if (error) {
                log_1.getLogger('AzureStorageClientUtility')
                    .error(`Create directory failed:, ${error}`);
                deferred.resolve(false);
            }
            else {
                deferred.resolve(true);
            }
        });
        return deferred.promise;
    }
    AzureStorageClientUtility.createDirectory = createDirectory;
    async function createDirectoryRecursive(fileServerClient, azureDirectory, azureShare) {
        const deferred = new ts_deferred_1.Deferred();
        const directories = azureDirectory.split('/');
        let rootDirectory = '';
        for (const directory of directories) {
            rootDirectory += directory;
            const result = await createDirectory(fileServerClient, rootDirectory, azureShare);
            if (!result) {
                deferred.resolve(false);
                return deferred.promise;
            }
            rootDirectory += '/';
        }
        deferred.resolve(true);
        return deferred.promise;
    }
    AzureStorageClientUtility.createDirectoryRecursive = createDirectoryRecursive;
    async function uploadFileToAzure(fileServerClient, azureDirectory, azureFileName, azureShare, localFilePath) {
        const deferred = new ts_deferred_1.Deferred();
        await fileServerClient.createFileFromLocalFile(azureShare, azureDirectory, azureFileName, localFilePath, (error, _result, _response) => {
            if (error) {
                log_1.getLogger('AzureStorageClientUtility')
                    .error(`Upload file failed:, ${error}`);
                deferred.resolve(false);
            }
            else {
                deferred.resolve(true);
            }
        });
        return deferred.promise;
    }
    async function downloadFile(fileServerClient, azureDirectory, azureFileName, azureShare, localFilePath) {
        const deferred = new ts_deferred_1.Deferred();
        await fileServerClient.getFileToStream(azureShare, azureDirectory, azureFileName, fs_1.default.createWriteStream(localFilePath), (error, _result, _response) => {
            if (error) {
                log_1.getLogger('AzureStorageClientUtility')
                    .error(`Download file failed:, ${error}`);
                deferred.resolve(false);
            }
            else {
                deferred.resolve(true);
            }
        });
        return deferred.promise;
    }
    async function uploadDirectory(fileServerClient, azureDirectory, azureShare, localDirectory) {
        const deferred = new ts_deferred_1.Deferred();
        const fileNameArray = fs_1.default.readdirSync(localDirectory);
        const result = await createDirectoryRecursive(fileServerClient, azureDirectory, azureShare);
        if (!result) {
            deferred.resolve(false);
            return deferred.promise;
        }
        for (const fileName of fileNameArray) {
            const fullFilePath = path_1.default.join(localDirectory, fileName);
            try {
                let resultUploadFile = true;
                let resultUploadDir = true;
                if (fs_1.default.lstatSync(fullFilePath)
                    .isFile()) {
                    resultUploadFile = await uploadFileToAzure(fileServerClient, azureDirectory, fileName, azureShare, fullFilePath);
                }
                else {
                    resultUploadDir = await uploadDirectory(fileServerClient, typescript_string_operations_1.String.Format('{0}/{1}', azureDirectory, fileName), azureShare, fullFilePath);
                }
                if (!(resultUploadFile && resultUploadDir)) {
                    deferred.resolve(false);
                    return deferred.promise;
                }
            }
            catch (error) {
                deferred.resolve(false);
                return deferred.promise;
            }
        }
        deferred.resolve(true);
        return deferred.promise;
    }
    AzureStorageClientUtility.uploadDirectory = uploadDirectory;
    async function downloadDirectory(fileServerClient, azureDirectory, azureShare, localDirectory) {
        const deferred = new ts_deferred_1.Deferred();
        await utils_1.mkDirP(localDirectory);
        fileServerClient.listFilesAndDirectoriesSegmented(azureShare, azureDirectory, 'null', async (_error, result, _response) => {
            if (('entries' in result) === false) {
                log_1.getLogger('AzureStorageClientUtility')
                    .error(`list files failed, can't get entries in result`);
                throw new Error(`list files failed, can't get entries in result`);
            }
            if (('files' in result.entries) === false) {
                log_1.getLogger('AzureStorageClientUtility')
                    .error(`list files failed, can't get files in result['entries']`);
                throw new Error(`list files failed, can't get files in result['entries']`);
            }
            if (('directories' in result.directories) === false) {
                log_1.getLogger('AzureStorageClientUtility')
                    .error(`list files failed, can't get directories in result['entries']`);
                throw new Error(`list files failed, can't get directories in result['entries']`);
            }
            for (const fileName of result.entries.files) {
                const fullFilePath = path_1.default.join(localDirectory, fileName.name);
                await downloadFile(fileServerClient, azureDirectory, fileName.name, azureShare, fullFilePath);
            }
            for (const directoryName of result.entries.directories) {
                const fullDirectoryPath = path_1.default.join(localDirectory, directoryName.name);
                const fullAzureDirectory = path_1.default.join(azureDirectory, directoryName.name);
                await downloadDirectory(fileServerClient, fullAzureDirectory, azureShare, fullDirectoryPath);
            }
            deferred.resolve();
        });
        return deferred.promise;
    }
    AzureStorageClientUtility.downloadDirectory = downloadDirectory;
})(AzureStorageClientUtility = exports.AzureStorageClientUtility || (exports.AzureStorageClientUtility = {}));
