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

	"github.com/weaviate/weaviate/entities/models"
)

// NewObjectsClassReferencesDeleteParams creates a new ObjectsClassReferencesDeleteParams object,
// with the default timeout for this client.
//
// Default values are not hydrated, since defaults are normally applied by the API server side.
//
// To enforce default values in parameter, use SetDefaults or WithDefaults.
func NewObjectsClassReferencesDeleteParams() *ObjectsClassReferencesDeleteParams {
	return &ObjectsClassReferencesDeleteParams{
		timeout: cr.DefaultTimeout,
	}
}

// NewObjectsClassReferencesDeleteParamsWithTimeout creates a new ObjectsClassReferencesDeleteParams object
// with the ability to set a timeout on a request.
func NewObjectsClassReferencesDeleteParamsWithTimeout(timeout time.Duration) *ObjectsClassReferencesDeleteParams {
	return &ObjectsClassReferencesDeleteParams{
		timeout: timeout,
	}
}

// NewObjectsClassReferencesDeleteParamsWithContext creates a new ObjectsClassReferencesDeleteParams object
// with the ability to set a context for a request.
func NewObjectsClassReferencesDeleteParamsWithContext(ctx context.Context) *ObjectsClassReferencesDeleteParams {
	return &ObjectsClassReferencesDeleteParams{
		Context: ctx,
	}
}

// NewObjectsClassReferencesDeleteParamsWithHTTPClient creates a new ObjectsClassReferencesDeleteParams object
// with the ability to set a custom HTTPClient for a request.
func NewObjectsClassReferencesDeleteParamsWithHTTPClient(client *http.Client) *ObjectsClassReferencesDeleteParams {
	return &ObjectsClassReferencesDeleteParams{
		HTTPClient: client,
	}
}

/*
ObjectsClassReferencesDeleteParams contains all the parameters to send to the API endpoint

	for the objects class references delete operation.

	Typically these are written to a http.Request.
*/
type ObjectsClassReferencesDeleteParams struct {

	// Body.
	Body *models.SingleRef

	/* ClassName.

	   The class name as defined in the schema
	*/
	ClassName string

	/* ConsistencyLevel.

	   Determines how many replicas must acknowledge a request before it is considered successful
	*/
	ConsistencyLevel *string

	/* ID.

	   Unique ID of the Object.

	   Format: uuid
	*/
	ID strfmt.UUID

	/* PropertyName.

	   Unique name of the property related to the Object.
	*/
	PropertyName string

	timeout    time.Duration
	Context    context.Context
	HTTPClient *http.Client
}

// WithDefaults hydrates default values in the objects class references delete params (not the query body).
//
// All values with no default are reset to their zero value.
func (o *ObjectsClassReferencesDeleteParams) WithDefaults() *ObjectsClassReferencesDeleteParams {
	o.SetDefaults()
	return o
}

// SetDefaults hydrates default values in the objects class references delete params (not the query body).
//
// All values with no default are reset to their zero value.
func (o *ObjectsClassReferencesDeleteParams) SetDefaults() {
	// no default values defined for this parameter
}

// WithTimeout adds the timeout to the objects class references delete params
func (o *ObjectsClassReferencesDeleteParams) WithTimeout(timeout time.Duration) *ObjectsClassReferencesDeleteParams {
	o.SetTimeout(timeout)
	return o
}

// SetTimeout adds the timeout to the objects class references delete params
func (o *ObjectsClassReferencesDeleteParams) SetTimeout(timeout time.Duration) {
	o.timeout = timeout
}

// WithContext adds the context to the objects class references delete params
func (o *ObjectsClassReferencesDeleteParams) WithContext(ctx context.Context) *ObjectsClassReferencesDeleteParams {
	o.SetContext(ctx)
	return o
}

// SetContext adds the context to the objects class references delete params
func (o *ObjectsClassReferencesDeleteParams) SetContext(ctx context.Context) {
	o.Context = ctx
}

// WithHTTPClient adds the HTTPClient to the objects class references delete params
func (o *ObjectsClassReferencesDeleteParams) WithHTTPClient(client *http.Client) *ObjectsClassReferencesDeleteParams {
	o.SetHTTPClient(client)
	return o
}

// SetHTTPClient adds the HTTPClient to the objects class references delete params
func (o *ObjectsClassReferencesDeleteParams) SetHTTPClient(client *http.Client) {
	o.HTTPClient = client
}

// WithBody adds the body to the objects class references delete params
func (o *ObjectsClassReferencesDeleteParams) WithBody(body *models.SingleRef) *ObjectsClassReferencesDeleteParams {
	o.SetBody(body)
	return o
}

// SetBody adds the body to the objects class references delete params
func (o *ObjectsClassReferencesDeleteParams) SetBody(body *models.SingleRef) {
	o.Body = body
}

// WithClassName adds the className to the objects class references delete params
func (o *ObjectsClassReferencesDeleteParams) WithClassName(className string) *ObjectsClassReferencesDeleteParams {
	o.SetClassName(className)
	return o
}

// SetClassName adds the className to the objects class references delete params
func (o *ObjectsClassReferencesDeleteParams) SetClassName(className string) {
	o.ClassName = className
}

// WithConsistencyLevel adds the consistencyLevel to the objects class references delete params
func (o *ObjectsClassReferencesDeleteParams) WithConsistencyLevel(consistencyLevel *string) *ObjectsClassReferencesDeleteParams {
	o.SetConsistencyLevel(consistencyLevel)
	return o
}

// SetConsistencyLevel adds the consistencyLevel to the objects class references delete params
func (o *ObjectsClassReferencesDeleteParams) SetConsistencyLevel(consistencyLevel *string) {
	o.ConsistencyLevel = consistencyLevel
}

// WithID adds the id to the objects class references delete params
func (o *ObjectsClassReferencesDeleteParams) WithID(id strfmt.UUID) *ObjectsClassReferencesDeleteParams {
	o.SetID(id)
	return o
}

// SetID adds the id to the objects class references delete params
func (o *ObjectsClassReferencesDeleteParams) SetID(id strfmt.UUID) {
	o.ID = id
}

// WithPropertyName adds the propertyName to the objects class references delete params
func (o *ObjectsClassReferencesDeleteParams) WithPropertyName(propertyName string) *ObjectsClassReferencesDeleteParams {
	o.SetPropertyName(propertyName)
	return o
}

// SetPropertyName adds the propertyName to the objects class references delete params
func (o *ObjectsClassReferencesDeleteParams) SetPropertyName(propertyName string) {
	o.PropertyName = propertyName
}

// WriteToRequest writes these params to a swagger request
func (o *ObjectsClassReferencesDeleteParams) WriteToRequest(r runtime.ClientRequest, reg strfmt.Registry) error {

	if err := r.SetTimeout(o.timeout); err != nil {
		return err
	}
	var res []error
	if o.Body != nil {
		if err := r.SetBodyParam(o.Body); err != nil {
			return err
		}
	}

	// path param className
	if err := r.SetPathParam("className", o.ClassName); err != nil {
		return err
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

	// path param id
	if err := r.SetPathParam("id", o.ID.String()); err != nil {
		return err
	}

	// path param propertyName
	if err := r.SetPathParam("propertyName", o.PropertyName); err != nil {
		return err
	}

	if len(res) > 0 {
		return errors.CompositeValidationError(res...)
	}
	return nil
}
