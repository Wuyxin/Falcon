"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.AdlTrialConfig = exports.NFSConfig = exports.ImagePullSecretConfig = exports.CheckpointConfig = void 0;
const kubernetesConfig_1 = require("../kubernetesConfig");
class CheckpointConfig {
    storageClass;
    storageSize;
    constructor(storageClass, storageSize) {
        this.storageClass = storageClass;
        this.storageSize = storageSize;
    }
}
exports.CheckpointConfig = CheckpointConfig;
class ImagePullSecretConfig {
    name;
    constructor(name) {
        this.name = name;
    }
}
exports.ImagePullSecretConfig = ImagePullSecretConfig;
class NFSConfig {
    server;
    path;
    containerMountPath;
    constructor(server, path, containerMountPath) {
        this.server = server;
        this.path = path;
        this.containerMountPath = containerMountPath;
    }
}
exports.NFSConfig = NFSConfig;
class AdlTrialConfig extends kubernetesConfig_1.KubernetesTrialConfig {
    command;
    gpuNum;
    image;
    namespace;
    imagePullSecrets;
    nfs;
    checkpoint;
    cpuNum;
    memorySize;
    adaptive;
    constructor(codeDir, command, gpuNum, image, namespace, imagePullSecrets, nfs, checkpoint, cpuNum, memorySize, adaptive) {
        super(codeDir);
        this.command = command;
        this.gpuNum = gpuNum;
        this.image = image;
        this.namespace = namespace;
        this.imagePullSecrets = imagePullSecrets;
        this.nfs = nfs;
        this.checkpoint = checkpoint;
        this.cpuNum = cpuNum;
        this.memorySize = memorySize;
        this.adaptive = adaptive;
    }
}
exports.AdlTrialConfig = AdlTrialConfig;
