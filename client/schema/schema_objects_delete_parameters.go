// Code generated by go-swagger; DO NOT EDIT.

package schema

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
	"github.com/go-openapi/swag"
)

// NewSchemaObjectsDeleteParams creates a new SchemaObjectsDeleteParams object,
// with the default timeout for this client.
//
// Default values are not hydrated, since defaults are normally applied by the API server side.
//
// To enforce default values in parameter, use SetDefaults or WithDefaults.
func NewSchemaObjectsDeleteParams() *SchemaObjectsDeleteParams {
	return &SchemaObjectsDeleteParams{
		timeout: cr.DefaultTimeout,
	}
}

// NewSchemaObjectsDeleteParamsWithTimeout creates a new SchemaObjectsDeleteParams object
// with the ability to set a timeout on a request.
func NewSchemaObjectsDeleteParamsWithTimeout(timeout time.Duration) *SchemaObjectsDeleteParams {
	return &SchemaObjectsDeleteParams{
		timeout: timeout,
	}
}

// NewSchemaObjectsDeleteParamsWithContext creates a new SchemaObjectsDeleteParams object
// with the ability to set a context for a request.
func NewSchemaObjectsDeleteParamsWithContext(ctx context.Context) *SchemaObjectsDeleteParams {
	return &SchemaObjectsDeleteParams{
		Context: ctx,
	}
}

// NewSchemaObjectsDeleteParamsWithHTTPClient creates a new SchemaObjectsDeleteParams object
// with the ability to set a custom HTTPClient for a request.
func NewSchemaObjectsDeleteParamsWithHTTPClient(client *http.Client) *SchemaObjectsDeleteParams {
	return &SchemaObjectsDeleteParams{
		HTTPClient: client,
	}
}

/*
SchemaObjectsDeleteParams contains all the parameters to send to the API endpoint

	for the schema objects delete operation.

	Typically these are written to a http.Request.
*/
type SchemaObjectsDeleteParams struct {

	// ClassName.
	ClassName string

	// Force.
	Force *bool

	timeout    time.Duration
	Context    context.Context
	HTTPClient *http.Client
}

// WithDefaults hydrates default values in the schema objects delete params (not the query body).
//
// All values with no default are reset to their zero value.
func (o *SchemaObjectsDeleteParams) WithDefaults() *SchemaObjectsDeleteParams {
	o.SetDefaults()
	return o
}

// SetDefaults hydrates default values in the schema objects delete params (not the query body).
//
// All values with no default are reset to their zero value.
func (o *SchemaObjectsDeleteParams) SetDefaults() {
	// no default values defined for this parameter
}

// WithTimeout adds the timeout to the schema objects delete params
func (o *SchemaObjectsDeleteParams) WithTimeout(timeout time.Duration) *SchemaObjectsDeleteParams {
	o.SetTimeout(timeout)
	return o
}

// SetTimeout adds the timeout to the schema objects delete params
func (o *SchemaObjectsDeleteParams) SetTimeout(timeout time.Duration) {
	o.timeout = timeout
}

// WithContext adds the context to the schema objects delete params
func (o *SchemaObjectsDeleteParams) WithContext(ctx context.Context) *SchemaObjectsDeleteParams {
	o.SetContext(ctx)
	return o
}

// SetContext adds the context to the schema objects delete params
func (o *SchemaObjectsDeleteParams) SetContext(ctx context.Context) {
	o.Context = ctx
}

// WithHTTPClient adds the HTTPClient to the schema objects delete params
func (o *SchemaObjectsDeleteParams) WithHTTPClient(client *http.Client) *SchemaObjectsDeleteParams {
	o.SetHTTPClient(client)
	return o
}

// SetHTTPClient adds the HTTPClient to the schema objects delete params
func (o *SchemaObjectsDeleteParams) SetHTTPClient(client *http.Client) {
	o.HTTPClient = client
}

// WithClassName adds the className to the schema objects delete params
func (o *SchemaObjectsDeleteParams) WithClassName(className string) *SchemaObjectsDeleteParams {
	o.SetClassName(className)
	return o
}

// SetClassName adds the className to the schema objects delete params
func (o *SchemaObjectsDeleteParams) SetClassName(className string) {
	o.ClassName = className
}

// WithForce adds the force to the schema objects delete params
func (o *SchemaObjectsDeleteParams) WithForce(force *bool) *SchemaObjectsDeleteParams {
	o.SetForce(force)
	return o
}

// SetForce adds the force to the schema objects delete params
func (o *SchemaObjectsDeleteParams) SetForce(force *bool) {
	o.Force = force
}

// WriteToRequest writes these params to a swagger request
func (o *SchemaObjectsDeleteParams) WriteToRequest(r runtime.ClientRequest, reg strfmt.Registry) error {

	if err := r.SetTimeout(o.timeout); err != nil {
		return err
	}
	var res []error

	// path param className
	if err := r.SetPathParam("className", o.ClassName); err != nil {
		return err
	}

	if o.Force != nil {

		// query param force
		var qrForce bool

		if o.Force != nil {
			qrForce = *o.Force
		}
		qForce := swag.FormatBool(qrForce)
		if qForce != "" {

			if err := r.SetQueryParam("force", qForce); err != nil {
				return err
			}
		}
	}

	if len(res) > 0 {
		return errors.CompositeValidationError(res...)
	}
	return nil
}
