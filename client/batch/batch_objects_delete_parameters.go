// Code generated by go-swagger; DO NOT EDIT.

package batch

// This file was generated by the swagger tool.
// Editing this file might prove futile when you re-run the swagger generate command

import (
	"context"
	"net/http"
	"time"

	"github.com/go-openapi/errors"
	"github.com/go-openapi/runtime"
	cr "github.com/go-openapi/runtime/client"
	"github.com/go-openapi/strfmt"

	"github.com/weaviate/weaviate/entities/models"
)

// NewBatchObjectsDeleteParams creates a new BatchObjectsDeleteParams object,
// with the default timeout for this client.
//
// Default values are not hydrated, since defaults are normally applied by the API server side.
//
// To enforce default values in parameter, use SetDefaults or WithDefaults.
func NewBatchObjectsDeleteParams() *BatchObjectsDeleteParams {
	return &BatchObjectsDeleteParams{
		timeout: cr.DefaultTimeout,
	}
}

// NewBatchObjectsDeleteParamsWithTimeout creates a new BatchObjectsDeleteParams object
// with the ability to set a timeout on a request.
func NewBatchObjectsDeleteParamsWithTimeout(timeout time.Duration) *BatchObjectsDeleteParams {
	return &BatchObjectsDeleteParams{
		timeout: timeout,
	}
}

// NewBatchObjectsDeleteParamsWithContext creates a new BatchObjectsDeleteParams object
// with the ability to set a context for a request.
func NewBatchObjectsDeleteParamsWithContext(ctx context.Context) *BatchObjectsDeleteParams {
	return &BatchObjectsDeleteParams{
		Context: ctx,
	}
}

// NewBatchObjectsDeleteParamsWithHTTPClient creates a new BatchObjectsDeleteParams object
// with the ability to set a custom HTTPClient for a request.
func NewBatchObjectsDeleteParamsWithHTTPClient(client *http.Client) *BatchObjectsDeleteParams {
	return &BatchObjectsDeleteParams{
		HTTPClient: client,
	}
}

/*
BatchObjectsDeleteParams contains all the parameters to send to the API endpoint

	for the batch objects delete operation.

	Typically these are written to a http.Request.
*/
type BatchObjectsDeleteParams struct {

	// Body.
	Body *models.BatchDelete

	/* ConsistencyLevel.

	   Determines how many replicas must acknowledge a request before it is considered successful
	*/
	ConsistencyLevel *string

	timeout    time.Duration
	Context    context.Context
	HTTPClient *http.Client
}

// WithDefaults hydrates default values in the batch objects delete params (not the query body).
//
// All values with no default are reset to their zero value.
func (o *BatchObjectsDeleteParams) WithDefaults() *BatchObjectsDeleteParams {
	o.SetDefaults()
	return o
}

// SetDefaults hydrates default values in the batch objects delete params (not the query body).
//
// All values with no default are reset to their zero value.
func (o *BatchObjectsDeleteParams) SetDefaults() {
	// no default values defined for this parameter
}

// WithTimeout adds the timeout to the batch objects delete params
func (o *BatchObjectsDeleteParams) WithTimeout(timeout time.Duration) *BatchObjectsDeleteParams {
	o.SetTimeout(timeout)
	return o
}

// SetTimeout adds the timeout to the batch objects delete params
func (o *BatchObjectsDeleteParams) SetTimeout(timeout time.Duration) {
	o.timeout = timeout
}

// WithContext adds the context to the batch objects delete params
func (o *BatchObjectsDeleteParams) WithContext(ctx context.Context) *BatchObjectsDeleteParams {
	o.SetContext(ctx)
	return o
}

// SetContext adds the context to the batch objects delete params
func (o *BatchObjectsDeleteParams) SetContext(ctx context.Context) {
	o.Context = ctx
}

// WithHTTPClient adds the HTTPClient to the batch objects delete params
func (o *BatchObjectsDeleteParams) WithHTTPClient(client *http.Client) *BatchObjectsDeleteParams {
	o.SetHTTPClient(client)
	return o
}

// SetHTTPClient adds the HTTPClient to the batch objects delete params
func (o *BatchObjectsDeleteParams) SetHTTPClient(client *http.Client) {
	o.HTTPClient = client
}

// WithBody adds the body to the batch objects delete params
func (o *BatchObjectsDeleteParams) WithBody(body *models.BatchDelete) *BatchObjectsDeleteParams {
	o.SetBody(body)
	return o
}

// SetBody adds the body to the batch objects delete params
func (o *BatchObjectsDeleteParams) SetBody(body *models.BatchDelete) {
	o.Body = body
}

// WithConsistencyLevel adds the consistencyLevel to the batch objects delete params
func (o *BatchObjectsDeleteParams) WithConsistencyLevel(consistencyLevel *string) *BatchObjectsDeleteParams {
	o.SetConsistencyLevel(consistencyLevel)
	return o
}

// SetConsistencyLevel adds the consistencyLevel to the batch objects delete params
func (o *BatchObjectsDeleteParams) SetConsistencyLevel(consistencyLevel *string) {
	o.ConsistencyLevel = consistencyLevel
}

// WriteToRequest writes these params to a swagger request
func (o *BatchObjectsDeleteParams) WriteToRequest(r runtime.ClientRequest, reg strfmt.Registry) error {

	if err := r.SetTimeout(o.timeout); err != nil {
		return err
	}
	var res []error
	if o.Body != nil {
		if err := r.SetBodyParam(o.Body); err != nil {
			return err
		}
	}

	if o.ConsistencyLevel != nil {

		// query param consistency_level
		var qrConsistencyLevel string

		if o.ConsistencyLevel != nil {
			qrConsistencyLevel = *o.ConsistencyLevel
		}
		qConsistencyLevel := qrConsistencyLevel
		if qConsistencyLevel != "" {

			if err := r.SetQueryParam("consistency_level", qConsistencyLevel); err != nil {
				return err
			}
		}
	}

	if len(res) > 0 {
		return errors.CompositeValidationError(res...)
	}
	return nil
}
