//                           _       _
// __      _____  __ ___   ___  __ _| |_ ___
// \ \ /\ / / _ \/ _` \ \ / / |/ _` | __/ _ \
//  \ V  V /  __/ (_| |\ V /| | (_| | ||  __/
//   \_/\_/ \___|\__,_| \_/ |_|\__,_|\__\___|
//
//  Copyright © 2016 - 2023 Weaviate B.V. All rights reserved.
//
//  CONTACT: hello@weaviate.io
//

// Code generated by go-swagger; DO NOT EDIT.

package models

// This file was generated by the swagger tool.
// Editing this file might prove futile when you re-run the swagger generate command

import (
	"context"
	"encoding/json"

	"github.com/go-openapi/errors"
	"github.com/go-openapi/strfmt"
	"github.com/go-openapi/swag"
	"github.com/go-openapi/validate"
)

// Property property
//
// swagger:model Property
type Property struct {

	// Can be a reference to another type when it starts with a capital (for example Person), otherwise "string" or "int".
	DataType []string `json:"dataType"`

	// Description of the property.
	Description string `json:"description,omitempty"`

	// Optional. Should this property be indexed in the inverted index. Defaults to true. If you choose false, you will not be able to use this property in where filters. This property has no affect on vectorization decisions done by modules
	IndexInverted *bool `json:"indexInverted,omitempty"`

	// Configuration specific to modules this Weaviate instance has installed
	ModuleConfig interface{} `json:"moduleConfig,omitempty"`

	// Name of the property as URI relative to the schema URL.
	Name string `json:"name,omitempty"`

	// Determines tokenization of the property as separate words or whole field. Optional. Applies to string, string[], text and text[] data types. Allowed values are `word` (default) and `field` for string and string[], `word` (default) for text and text[]. Not supported for remaining data types
	// Enum: [word field]
	Tokenization string `json:"tokenization,omitempty"`
}

// Validate validates this property
func (m *Property) Validate(formats strfmt.Registry) error {
	var res []error

	if err := m.validateTokenization(formats); err != nil {
		res = append(res, err)
	}

	if len(res) > 0 {
		return errors.CompositeValidationError(res...)
	}
	return nil
}

var propertyTypeTokenizationPropEnum []interface{}

func init() {
	var res []string
	if err := json.Unmarshal([]byte(`["word","field"]`), &res); err != nil {
		panic(err)
	}
	for _, v := range res {
		propertyTypeTokenizationPropEnum = append(propertyTypeTokenizationPropEnum, v)
	}
}

const (

	// PropertyTokenizationWord captures enum value "word"
	PropertyTokenizationWord string = "word"

	// PropertyTokenizationField captures enum value "field"
	PropertyTokenizationField string = "field"
)

// prop value enum
func (m *Property) validateTokenizationEnum(path, location string, value string) error {
	if err := validate.EnumCase(path, location, value, propertyTypeTokenizationPropEnum, true); err != nil {
		return err
	}
	return nil
}

func (m *Property) validateTokenization(formats strfmt.Registry) error {
	if swag.IsZero(m.Tokenization) { // not required
		return nil
	}

	// value enum
	if err := m.validateTokenizationEnum("tokenization", "body", m.Tokenization); err != nil {
		return err
	}

	return nil
}

// ContextValidate validates this property based on context it is used
func (m *Property) ContextValidate(ctx context.Context, formats strfmt.Registry) error {
	return nil
}

// MarshalBinary interface implementation
func (m *Property) MarshalBinary() ([]byte, error) {
	if m == nil {
		return nil, nil
	}
	return swag.WriteJSON(m)
}

// UnmarshalBinary interface implementation
func (m *Property) UnmarshalBinary(b []byte) error {
	var res Property
	if err := swag.ReadJSON(b, &res); err != nil {
		return err
	}
	*m = res
	return nil
}
