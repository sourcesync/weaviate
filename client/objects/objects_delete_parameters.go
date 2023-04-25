// Code generated by go-swagger; DO NOT EDIT.

package objects

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
)

// NewObjectsDeleteParams creates a new ObjectsDeleteParams object,
// with the default timeout for this client.
//
// Default values are not hydrated, since defaults are normally applied by the API server side.
//
// To enforce default values in parameter, use SetDefaults or WithDefaults.
func NewObjectsDeleteParams() *ObjectsDeleteParams {
	return &ObjectsDeleteParams{
		timeout: cr.DefaultTimeout,
	}
}

// NewObjectsDeleteParamsWithTimeout creates a new ObjectsDeleteParams object
// with the ability to set a timeout on a request.
func NewObjectsDeleteParamsWithTimeout(timeout time.Duration) *ObjectsDeleteParams {
	return &ObjectsDeleteParams{
		timeout: timeout,
	}
}

// NewObjectsDeleteParamsWithContext creates a new ObjectsDeleteParams object
// with the ability to set a context for a request.
func NewObjectsDeleteParamsWithContext(ctx context.Context) *ObjectsDeleteParams {
	return &ObjectsDeleteParams{
		Context: ctx,
	}
}

// NewObjectsDeleteParamsWithHTTPClient creates a new ObjectsDeleteParams object
// with the ability to set a custom HTTPClient for a request.
func NewObjectsDeleteParamsWithHTTPClient(client *http.Client) *ObjectsDeleteParams {
	return &ObjectsDeleteParams{
		HTTPClient: client,
	}
}

/*
ObjectsDeleteParams contains all the parameters to send to the API endpoint

	for the objects delete operation.

	Typically these are written to a http.Request.
*/
type ObjectsDeleteParams struct {

	/* ConsistencyLevel.

	   Determines how many replicas must acknowledge a request before it is considered successful
	*/
	ConsistencyLevel *string

	/* ID.

	   Unique ID of the Object.

	   Format: uuid
	*/
	ID strfmt.UUID

	timeout    time.Duration
	Context    context.Context
	HTTPClient *http.Client
}

// WithDefaults hydrates default values in the objects delete params (not the query body).
//
// All values with no default are reset to their zero value.
func (o *ObjectsDeleteParams) WithDefaults() *ObjectsDeleteParams {
	o.SetDefaults()
	return o
}

// SetDefaults hydrates default values in the objects delete params (not the query body).
//
// All values with no default are reset to their zero value.
func (o *ObjectsDeleteParams) SetDefaults() {
	// no default values defined for this parameter
}

// WithTimeout adds the timeout to the objects delete params
func (o *ObjectsDeleteParams) WithTimeout(timeout time.Duration) *ObjectsDeleteParams {
	o.SetTimeout(timeout)
	return o
}

// SetTimeout adds the timeout to the objects delete params
func (o *ObjectsDeleteParams) SetTimeout(timeout time.Duration) {
	o.timeout = timeout
}

// WithContext adds the context to the objects delete params
func (o *ObjectsDeleteParams) WithContext(ctx context.Context) *ObjectsDeleteParams {
	o.SetContext(ctx)
	return o
}

// SetContext adds the context to the objects delete params
func (o *ObjectsDeleteParams) SetContext(ctx context.Context) {
	o.Context = ctx
}

// WithHTTPClient adds the HTTPClient to the objects delete params
func (o *ObjectsDeleteParams) WithHTTPClient(client *http.Client) *ObjectsDeleteParams {
	o.SetHTTPClient(client)
	return o
}

// SetHTTPClient adds the HTTPClient to the objects delete params
func (o *ObjectsDeleteParams) SetHTTPClient(client *http.Client) {
	o.HTTPClient = client
}

// WithConsistencyLevel adds the consistencyLevel to the objects delete params
func (o *ObjectsDeleteParams) WithConsistencyLevel(consistencyLevel *string) *ObjectsDeleteParams {
	o.SetConsistencyLevel(consistencyLevel)
	return o
}

// SetConsistencyLevel adds the consistencyLevel to the objects delete params
func (o *ObjectsDeleteParams) SetConsistencyLevel(consistencyLevel *string) {
	o.ConsistencyLevel = consistencyLevel
}

// WithID adds the id to the objects delete params
func (o *ObjectsDeleteParams) WithID(id strfmt.UUID) *ObjectsDeleteParams {
	o.SetID(id)
	return o
}

// SetID adds the id to the objects delete params
func (o *ObjectsDeleteParams) SetID(id strfmt.UUID) {
	o.ID = id
}

// WriteToRequest writes these params to a swagger request
func (o *ObjectsDeleteParams) WriteToRequest(r runtime.ClientRequest, reg strfmt.Registry) error {

	if err := r.SetTimeout(o.timeout); err != nil {
		return err
	}
	var res []error

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

	// path param id
	if err := r.SetPathParam("id", o.ID.String()); err != nil {
		return err
	}

	if len(res) > 0 {
		return errors.CompositeValidationError(res...)
	}
	return nil
}
