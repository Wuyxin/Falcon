"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.SqlDB = void 0;
const assert_1 = __importDefault(require("assert"));
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
const sqlite3_1 = __importDefault(require("sqlite3"));
const ts_deferred_1 = require("ts-deferred");
const log_1 = require("../common/log");
const createTables = `
create table TrialJobEvent (timestamp integer, trialJobId text, event text, data text, logPath text, sequenceId integer, message text);
create index TrialJobEvent_trialJobId on TrialJobEvent(trialJobId);
create index TrialJobEvent_event on TrialJobEvent(event);

create table MetricData (timestamp integer, trialJobId text, parameterId text, type text, sequence integer, data text);
create index MetricData_trialJobId on MetricData(trialJobId);
create index MetricData_type on MetricData(type);

create table ExperimentProfile (
    params text,
    id text,
    execDuration integer,
    startTime integer,
    endTime integer,
    logDir text,
    nextSequenceId integer,
    revision integer);
create index ExperimentProfile_id on ExperimentProfile(id);
`;
function loadExperimentProfile(row) {
    return {
        params: JSON.parse(row.params),
        id: row.id,
        execDuration: row.execDuration,
        startTime: row.startTime === null ? undefined : row.startTime,
        endTime: row.endTime === null ? undefined : row.endTime,
        logDir: row.logDir === null ? undefined : row.logDir,
        nextSequenceId: row.nextSequenceId,
        revision: row.revision
    };
}
function loadTrialJobEvent(row) {
    return {
        timestamp: row.timestamp,
        trialJobId: row.trialJobId,
        event: row.event,
        data: row.data === null ? undefined : row.data,
        logPath: row.logPath === null ? undefined : row.logPath,
        sequenceId: row.sequenceId === null ? undefined : row.sequenceId,
        message: row.message === null ? undefined : row.message
    };
}
function loadMetricData(row) {
    return {
        timestamp: row.timestamp,
        trialJobId: row.trialJobId,
        parameterId: row.parameterId,
        type: row.type,
        sequence: row.sequence,
        data: row.data
    };
}
class SqlDB {
    db;
    log = log_1.getLogger('SqlDB');
    initTask;
    init(createNew, dbDir) {
        if (this.initTask !== undefined) {
            return this.initTask.promise;
        }
        this.initTask = new ts_deferred_1.Deferred();
        this.log.debug(`Database directory: ${dbDir}`);
        assert_1.default(fs_1.default.existsSync(dbDir));
        const mode = createNew ? (sqlite3_1.default.OPEN_CREATE | sqlite3_1.default.OPEN_READWRITE) : sqlite3_1.default.OPEN_READWRITE;
        const dbFileName = path_1.default.join(dbDir, 'nni.sqlite');
        this.db = new sqlite3_1.default.Database(dbFileName, mode, (err) => {
            if (err) {
                this.resolve(this.initTask, err);
            }
            else {
                if (createNew) {
                    this.db.exec(createTables, (_error) => {
                        this.resolve(this.initTask, err);
                    });
                }
                else {
                    this.initTask.resolve();
                }
            }
        });
        return this.initTask.promise;
    }
    close() {
        const deferred = new ts_deferred_1.Deferred();
        this.db.close((err) => { this.resolve(deferred, err); });
        return deferred.promise;
    }
    storeExperimentProfile(exp) {
        const sql = 'insert into ExperimentProfile values (?,?,?,?,?,?,?,?)';
        const args = [
            JSON.stringify(exp.params),
            exp.id,
            exp.execDuration,
            exp.startTime === undefined ? null : exp.startTime,
            exp.endTime === undefined ? null : exp.endTime,
            exp.logDir === undefined ? null : exp.logDir,
            exp.nextSequenceId,
            exp.revision
        ];
        this.log.trace(`storeExperimentProfile: SQL: ${sql}, args:`, args);
        const deferred = new ts_deferred_1.Deferred();
        this.db.run(sql, args, (err) => { this.resolve(deferred, err); });
        return deferred.promise;
    }
    queryExperimentProfile(experimentId, revision) {
        let sql = '';
        let args = [];
        if (revision === undefined) {
            sql = 'select * from ExperimentProfile where id=? order by revision DESC';
            args = [experimentId];
        }
        else {
            sql = 'select * from ExperimentProfile where id=? and revision=?';
            args = [experimentId, revision];
        }
        this.log.trace(`queryExperimentProfile: SQL: ${sql}, args:`, args);
        const deferred = new ts_deferred_1.Deferred();
        this.db.all(sql, args, (err, rows) => {
            this.resolve(deferred, err, rows, loadExperimentProfile);
        });
        return deferred.promise;
    }
    async queryLatestExperimentProfile(experimentId) {
        const profiles = await this.queryExperimentProfile(experimentId);
        return profiles[0];
    }
    storeTrialJobEvent(event, trialJobId, timestamp, hyperParameter, jobDetail) {
        const sql = 'insert into TrialJobEvent values (?,?,?,?,?,?,?)';
        const logPath = jobDetail === undefined ? undefined : jobDetail.url;
        const sequenceId = jobDetail === undefined ? undefined : jobDetail.form.sequenceId;
        const message = jobDetail === undefined ? undefined : jobDetail.message;
        const args = [timestamp, trialJobId, event, hyperParameter, logPath, sequenceId, message];
        this.log.trace(`storeTrialJobEvent: SQL: ${sql}, args:`, args);
        const deferred = new ts_deferred_1.Deferred();
        this.db.run(sql, args, (err) => { this.resolve(deferred, err); });
        return deferred.promise;
    }
    queryTrialJobEvent(trialJobId, event) {
        let sql = '';
        let args;
        if (trialJobId === undefined && event === undefined) {
            sql = 'select * from TrialJobEvent';
        }
        else if (trialJobId === undefined) {
            sql = 'select * from TrialJobEvent where event=?';
            args = [event];
        }
        else if (event === undefined) {
            sql = 'select * from TrialJobEvent where trialJobId=?';
            args = [trialJobId];
        }
        else {
            sql = 'select * from TrialJobEvent where trialJobId=? and event=?';
            args = [trialJobId, event];
        }
        this.log.trace(`queryTrialJobEvent: SQL: ${sql}, args:`, args);
        const deferred = new ts_deferred_1.Deferred();
        this.db.all(sql, args, (err, rows) => {
            this.resolve(deferred, err, rows, loadTrialJobEvent);
        });
        return deferred.promise;
    }
    storeMetricData(_trialJobId, data) {
        const sql = 'insert into MetricData values (?,?,?,?,?,?)';
        const json = JSON.parse(data);
        const args = [Date.now(), json.trialJobId, json.parameterId, json.type, json.sequence, JSON.stringify(json.data)];
        this.log.trace(`storeMetricData: SQL: ${sql}, args:`, args);
        const deferred = new ts_deferred_1.Deferred();
        this.db.run(sql, args, (err) => { this.resolve(deferred, err); });
        return deferred.promise;
    }
    queryMetricData(trialJobId, metricType) {
        let sql = '';
        let args;
        if (metricType === undefined && trialJobId === undefined) {
            sql = 'select * from MetricData';
        }
        else if (trialJobId === undefined) {
            sql = 'select * from MetricData where type=?';
            args = [metricType];
        }
        else if (metricType === undefined) {
            sql = 'select * from MetricData where trialJobId=?';
            args = [trialJobId];
        }
        else {
            sql = 'select * from MetricData where trialJobId=? and type=?';
            args = [trialJobId, metricType];
        }
        this.log.trace(`queryMetricData: SQL: ${sql}, args:`, args);
        const deferred = new ts_deferred_1.Deferred();
        this.db.all(sql, args, (err, rows) => {
            this.resolve(deferred, err, rows, loadMetricData);
        });
        return deferred.promise;
    }
    resolve(deferred, error, rows, rowLoader) {
        if (error !== null) {
            deferred.reject(error);
            return;
        }
        if (rowLoader === undefined) {
            deferred.resolve();
        }
        else {
            const data = [];
            for (const row of rows) {
                data.push(rowLoader(row));
            }
            this.log.trace(`sql query result:`, data);
            deferred.resolve(data);
        }
    }
}
exports.SqlDB = SqlDB;
