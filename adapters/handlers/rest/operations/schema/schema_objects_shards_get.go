// Code generated by go-swagger; DO NOT EDIT.

package schema

// This file was generated by the swagger tool.
// Editing this file might prove futile when you re-run the generate command

import (
	"net/http"

	"github.com/go-openapi/runtime/middleware"

	"github.com/weaviate/weaviate/entities/models"
)

// SchemaObjectsShardsGetHandlerFunc turns a function with the right signature into a schema objects shards get handler
type SchemaObjectsShardsGetHandlerFunc func(SchemaObjectsShardsGetParams, *models.Principal) middleware.Responder

// Handle executing the request and returning a response
func (fn SchemaObjectsShardsGetHandlerFunc) Handle(params SchemaObjectsShardsGetParams, principal *models.Principal) middleware.Responder {
	return fn(params, principal)
}

// SchemaObjectsShardsGetHandler interface for that can handle valid schema objects shards get params
type SchemaObjectsShardsGetHandler interface {
	Handle(SchemaObjectsShardsGetParams, *models.Principal) middleware.Responder
}

// NewSchemaObjectsShardsGet creates a new http.Handler for the schema objects shards get operation
func NewSchemaObjectsShardsGet(ctx *middleware.Context, handler SchemaObjectsShardsGetHandler) *SchemaObjectsShardsGet {
	return &SchemaObjectsShardsGet{Context: ctx, Handler: handler}
}

/*
	SchemaObjectsShardsGet swagger:route GET /schema/{className}/shards schema schemaObjectsShardsGet

Get the shards status of an Object class
*/
type SchemaObjectsShardsGet struct {
	Context *middleware.Context
	Handler SchemaObjectsShardsGetHandler
}

func (o *SchemaObjectsShardsGet) ServeHTTP(rw http.ResponseWriter, r *http.Request) {
	route, rCtx, _ := o.Context.RouteInfo(r)
	if rCtx != nil {
		*r = *rCtx
	}
	var Params = NewSchemaObjectsShardsGetParams()
	uprinc, aCtx, err := o.Context.Authorize(r, route)
	if err != nil {
		o.Context.Respond(rw, r, route.Produces, route, err)
		return
	}
	if aCtx != nil {
		*r = *aCtx
	}
	var principal *models.Principal
	if uprinc != nil {
		principal = uprinc.(*models.Principal) // this is really a models.Principal, I promise
	}

	if err := o.Context.BindValidRequest(r, route, &Params); err != nil { // bind params
		o.Context.Respond(rw, r, route.Produces, route, err)
		return
	}

	res := o.Handler.Handle(Params, principal) // actually handle the request
	o.Context.Respond(rw, r, route.Produces, route, res)

}
