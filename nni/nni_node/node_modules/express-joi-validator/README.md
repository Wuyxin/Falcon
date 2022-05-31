express-joi-validator
===========

[![Build Status](https://travis-ci.org/Threadster/express-joi-validator.svg)](https://travis-ci.org/Threadster/express-joi-validator)

Simple validation middleware for express using the [Joi](https://github.com/spumko/joi) validation suite.

## Installation
```
npm install express-joi-validator
```

## Dependencies
```
npm install joi
```

## Usage
```javascript

var Express = require('express');
var BodyParser = require('body-parser');
var ExpressJoi = require('express-joi-validator');
var Joi = require('joi');

var app = Express();
app.use(BodyParser.json());

// Use Joi to create your schemas
var querySchema = {
    query: {
        limit: Joi.number().default(10).min(10).max(100),
        offset: Joi.number().default(10).min(10).max(100)
    }
};

// Attach the validator like other middleware
app.get('/', expressJoi(querySchema), function (req, res, next) {
   	// do something with req.query.limit & req.query.offset
    ...
});

// Use Joi to create your schemas
var bodySchema = {
    body: {
        name: Joi.string().required()
    }
};

// Attach the validator like other middleware
app.post('/', expressJoi(bodySchema), function (req, res, next) {
	// do something with req.body;
    ...
});

// Example error handler
app.use(function (err, req, res, next) {
    if (err.isBoom) {
         return res.status(err.output.statusCode).json(err.output.payload);
    }
});



app.listen(8080);
```
If a validation error occurs it will either be handled by your express error handling middleware or thrown.


## [Joi](https://github.com/spumko/joi)

Since this middleware is just a wrapper around the Joi [`validate`](https://github.com/hapijs/joi#validate) method all the [options](https://github.com/hapijs/joi#validatevalue-schema-options-callback)
are supported.


## Running Tests

```
npm install
```
```
npm test
```