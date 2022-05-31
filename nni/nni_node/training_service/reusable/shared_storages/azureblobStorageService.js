"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.AzureBlobSharedStorageService = void 0;
const child_process_promise_1 = __importDefault(require("child-process-promise"));
const path_1 = __importDefault(require("path"));
const sharedStorage_1 = require("../sharedStorage");
const mountedStorageService_1 = require("../storages/mountedStorageService");
const log_1 = require("common/log");
const experimentStartupInfo_1 = require("common/experimentStartupInfo");
const INSTALL_BLOBFUSE = `
#!/bin/bash
if [ -n "$(command -v blobfuse)" ]
then
    exit 0
fi

if [ -n "$(command -v apt-get)" ]
then
    sudo apt-get update
    sudo apt-get install -y lsb-release
elif [ -n "$(command -v yum)" ]
then
    sudo yum install -y redhat-lsb
else
    echo "Unknown package management."
    exit 1
fi

id=$(lsb_release -i | cut -c16- | sed s/[[:space:]]//g)
version=$(lsb_release -r | cut -c9- | sed s/[[:space:]]//g)

if [ "$id" = "Ubuntu" ]
then
    wget https://packages.microsoft.com/config/ubuntu/$version/packages-microsoft-prod.deb
    sudo DEBIAN_FRONTEND=noninteractive dpkg -i packages-microsoft-prod.deb
    sudo apt-get update
    sudo apt-get install -y blobfuse fuse
elif [ "$id" = "CentOS" ] || [ "$id" = "RHEL" ]
then
    sudo rpm -Uvh https://packages.microsoft.com/config/rhel/$(echo $version | cut -c1)/packages-microsoft-prod.rpm
    sudo yum install -y blobfuse fuse
else
    echo "Not support distributor."
    exit 1
fi
`;
class AzureBlobSharedStorageService extends sharedStorage_1.SharedStorageService {
    log;
    internalStorageService;
    experimentId;
    localMounted;
    storageType;
    storageAccountName;
    storageAccountKey;
    containerName;
    localMountPoint;
    remoteMountPoint;
    constructor() {
        super();
        this.log = log_1.getLogger('AzureBlobSharedStorageService');
        this.internalStorageService = new mountedStorageService_1.MountedStorageService();
        this.experimentId = experimentStartupInfo_1.getExperimentId();
    }
    async config(azureblobConfig) {
        this.localMountPoint = azureblobConfig.localMountPoint;
        this.remoteMountPoint = azureblobConfig.remoteMountPoint;
        this.storageType = azureblobConfig.storageType;
        this.storageAccountName = azureblobConfig.storageAccountName;
        this.containerName = azureblobConfig.containerName;
        if (azureblobConfig.storageAccountKey !== undefined) {
            this.storageAccountKey = azureblobConfig.storageAccountKey;
        }
        else {
            const errorMessage = `${this.storageType} Shared Storage: must set 'storageAccountKey'.`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }
        this.localMounted = azureblobConfig.localMounted;
        if (this.localMounted === 'nnimount') {
            await this.helpLocalMount();
        }
        else if (this.localMounted === 'nomount') {
            const errorMessage = `${this.storageType} Shared Storage: ${this.storageType} not Support 'nomount' yet.`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }
        if (this.canLocalMounted && this.localMountPoint) {
            this.internalStorageService.initialize(this.localMountPoint, path_1.default.join(this.localMountPoint, 'nni', this.experimentId));
        }
    }
    get canLocalMounted() {
        return true;
    }
    get storageService() {
        return this.internalStorageService;
    }
    get localMountCommand() {
        if (this.localMountPoint) {
            return this.getCommand(this.localMountPoint);
        }
        else {
            this.log.error(`${this.storageType} Shared Storage: localMountPoint is not initialized.`);
            return '';
        }
    }
    get remoteMountCommand() {
        if (this.remoteMountPoint) {
            return this.getCommand(this.remoteMountPoint);
        }
        else {
            this.log.error(`${this.storageType} Shared Storage: remoteMountPoint is not initialized.`);
            return '';
        }
    }
    get remoteUmountCommand() {
        if (this.remoteMountPoint) {
            return `sudo umount -l ${this.remoteMountPoint}`;
        }
        else {
            this.log.error(`${this.storageType} Shared Storage: remoteMountPoint is not initialized.`);
            return '';
        }
    }
    getCommand(mountPoint) {
        const install = `rm -f nni_install_fuseblob.sh && touch nni_install_fuseblob.sh && echo "${INSTALL_BLOBFUSE.replace(/\$/g, `\\$`).replace(/\n/g, `\\n`).replace(/"/g, `\\"`)}" >> nni_install_fuseblob.sh && bash nni_install_fuseblob.sh`;
        const prepare = `sudo mkdir /mnt/resource/nniblobfusetmp -p && rm -f nni_fuse_connection.cfg && touch nni_fuse_connection.cfg && echo "accountName ${this.storageAccountName}\\naccountKey ${this.storageAccountKey}\\ncontainerName ${this.containerName}" >> nni_fuse_connection.cfg`;
        const mount = `mkdir -p ${mountPoint} && sudo blobfuse ${mountPoint} --tmp-path=/mnt/resource/nniblobfusetmp  --config-file=$(pwd)/nni_fuse_connection.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120 -o allow_other`;
        const clean = `rm -f nni_install_fuseblob.sh nni_fuse_connection.cfg`;
        return `${install} && ${prepare} && ${mount} && ${clean}`;
    }
    get localWorkingRoot() {
        return `${this.localMountPoint}/nni/${this.experimentId}`;
    }
    get remoteWorkingRoot() {
        return `${this.remoteMountPoint}/nni/${this.experimentId}`;
    }
    async helpLocalMount() {
        if (process.platform === 'win32') {
            const errorMessage = `${this.storageType} Shared Storage: ${this.storageType} do not support mount under Windows yet.`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }
        try {
            this.log.debug(`Local mount command is: ${this.localMountCommand}`);
            const result = await child_process_promise_1.default.exec(this.localMountCommand);
            if (result.stderr) {
                throw new Error(result.stderr);
            }
        }
        catch (error) {
            const errorMessage = `${this.storageType} Shared Storage: Mount ${this.storageAccountName}/${this.containerName} to ${this.localMountPoint} failed, error is ${error}`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }
        return Promise.resolve();
    }
    async cleanUp() {
        if (this.localMounted !== 'nnimount') {
            return Promise.resolve();
        }
        try {
            const result = await child_process_promise_1.default.exec(`sudo umount -l ${this.localMountPoint}`);
            if (result.stderr) {
                throw new Error(result.stderr);
            }
        }
        catch (error) {
            const errorMessage = `${this.storageType} Shared Storage: Umount ${this.localMountPoint}  failed, error is ${error}`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }
        return Promise.resolve();
    }
}
exports.AzureBlobSharedStorageService = AzureBlobSharedStorageService;
