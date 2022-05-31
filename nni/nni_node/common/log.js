"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.stopLogging = exports.startLogging = exports.setLogLevel = exports.getLogger = exports.Logger = exports.FATAL = exports.TRACE = exports.CRITICAL = exports.ERROR = exports.WARNING = exports.INFO = exports.DEBUG = void 0;
const fs_1 = __importDefault(require("fs"));
const util_1 = __importDefault(require("util"));
exports.DEBUG = 10;
exports.INFO = 20;
exports.WARNING = 30;
exports.ERROR = 40;
exports.CRITICAL = 50;
exports.TRACE = 1;
exports.FATAL = 50;
const levelNames = new Map([
    [exports.CRITICAL, 'CRITICAL'],
    [exports.ERROR, 'ERROR'],
    [exports.WARNING, 'WARNING'],
    [exports.INFO, 'INFO'],
    [exports.DEBUG, 'DEBUG'],
    [exports.TRACE, 'TRACE'],
]);
let logLevel = 0;
const loggers = new Map();
class Logger {
    name;
    constructor(name = 'root') {
        this.name = name;
    }
    trace(...args) {
        this.log(exports.TRACE, args);
    }
    debug(...args) {
        this.log(exports.DEBUG, args);
    }
    info(...args) {
        this.log(exports.INFO, args);
    }
    warning(...args) {
        this.log(exports.WARNING, args);
    }
    error(...args) {
        this.log(exports.ERROR, args);
    }
    critical(...args) {
        this.log(exports.CRITICAL, args);
    }
    fatal(...args) {
        this.log(exports.FATAL, args);
    }
    log(level, args) {
        const logFile = global.logFile;
        if (level < logLevel) {
            return;
        }
        const zeroPad = (num) => num.toString().padStart(2, '0');
        const now = new Date();
        const date = now.getFullYear() + '-' + zeroPad(now.getMonth() + 1) + '-' + zeroPad(now.getDate());
        const time = zeroPad(now.getHours()) + ':' + zeroPad(now.getMinutes()) + ':' + zeroPad(now.getSeconds());
        const datetime = date + ' ' + time;
        const levelName = levelNames.has(level) ? levelNames.get(level) : level.toString();
        const message = args.map(arg => (typeof arg === 'string' ? arg : util_1.default.inspect(arg))).join(' ');
        const record = `[${datetime}] ${levelName} (${this.name}) ${message}`;
        if (logFile === undefined) {
            if (!isUnitTest()) {
                console.log(record);
            }
        }
        else {
            logFile.write(record + '\n');
        }
    }
}
exports.Logger = Logger;
function getLogger(name = 'root') {
    let logger = loggers.get(name);
    if (logger === undefined) {
        logger = new Logger(name);
        loggers.set(name, logger);
    }
    return logger;
}
exports.getLogger = getLogger;
function setLogLevel(levelName) {
    if (levelName) {
        const level = module.exports[levelName.toUpperCase()];
        if (typeof level === 'number') {
            logLevel = level;
        }
        else {
            console.log('[ERROR] Bad log level:', levelName);
            getLogger('logging').error('Bad log level:', levelName);
        }
    }
}
exports.setLogLevel = setLogLevel;
function startLogging(logPath) {
    global.logFile = fs_1.default.createWriteStream(logPath, {
        flags: 'a+',
        encoding: 'utf8',
        autoClose: true
    });
}
exports.startLogging = startLogging;
function stopLogging() {
    if (global.logFile !== undefined) {
        global.logFile.end();
        global.logFile = undefined;
    }
}
exports.stopLogging = stopLogging;
function isUnitTest() {
    const event = process.env['npm_lifecycle_event'] ?? '';
    return event.startsWith('test') || event === 'mocha' || event === 'nyc';
}
