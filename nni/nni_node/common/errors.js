"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.MethodNotImplementedError = exports.NNIError = exports.NNIErrorNames = void 0;
var NNIErrorNames;
(function (NNIErrorNames) {
    NNIErrorNames.NOT_FOUND = 'NOT_FOUND';
    NNIErrorNames.INVALID_JOB_DETAIL = 'NO_VALID_JOB_DETAIL_FOUND';
    NNIErrorNames.RESOURCE_NOT_AVAILABLE = 'RESOURCE_NOT_AVAILABLE';
})(NNIErrorNames = exports.NNIErrorNames || (exports.NNIErrorNames = {}));
class NNIError extends Error {
    cause;
    constructor(name, message, err) {
        super(message);
        this.name = name;
        if (err !== undefined) {
            this.stack = err.stack;
        }
        this.cause = err;
    }
    static FromError(err, messagePrefix) {
        const msgPrefix = messagePrefix === undefined ? '' : messagePrefix;
        if (err instanceof NNIError) {
            if (err.message !== undefined) {
                err.message = msgPrefix + err.message;
            }
            return err;
        }
        else if (typeof (err) === 'string') {
            return new NNIError('', msgPrefix + err);
        }
        else if (err instanceof Error) {
            return new NNIError('', msgPrefix + err.message, err);
        }
        else {
            throw new Error(`Wrong instance type: ${typeof (err)}`);
        }
    }
}
exports.NNIError = NNIError;
class MethodNotImplementedError extends Error {
    constructor() {
        super('Method not implemented.');
    }
}
exports.MethodNotImplementedError = MethodNotImplementedError;
