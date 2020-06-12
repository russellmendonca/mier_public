# Copyright 2018 Google LLC
# Modifications copyright 2018 Xinyang Geng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator
from collections import OrderedDict

import tensorflow as tf


class ParameterizedModel(object):
    """Automatically parameterize model with one vector."""

    def __init__(self, name):
        self._name = name
        self._template_variable_scope = None
        self._parameter_scope = None
        self._parameterized_variable_scopes = []

    def build_template(self, *args, **kwargs):
        # Make sure this method has not been called before.
        assert self._template_variable_scope is None
        self._template_variable_scope = TemplateVariableScope(
            name=self.template_scope_name,
            exit_callback=self._build_parameter,
            *args, **kwargs
        )
        return self._template_variable_scope

    def _build_parameter(self):
        assert self._template_variable_scope is not None
        assert self._parameter_scope is None
        self._parameter_scope = ParameterizedVariableScope(
                self._template_variable_scope,
                name=self.parameterized_scope_name + '_parameter',
                parameter=None
        )
        with self._parameter_scope:
            pass
        return self._parameter_scope.parameter

    def build_parameterized(self, parameter=None, *args, **kwargs):
        # Make sure the template is already built.
        assert self._template_variable_scope is not None
        assert self._parameter_scope is not None
        if parameter is None:
            parameter = self.parameter

        parameterized_scope = ParameterizedVariableScope(
            self._template_variable_scope,
            name=self.parameterized_scope_name + '_{}'.format(len(self)),
            parameter=parameter,
            *args, **kwargs
        )
        self.parameterized_variable_scopes.append(parameterized_scope)
        return parameterized_scope

    @property
    def template_scope_name(self):
        return self.name + '_template'

    @property
    def parameterized_scope_name(self):
        return self.name + '_parameterized'

    @property
    def template_variable_scope(self):
        return self._template_variable_scope

    @property
    def parameterized_variable_scopes(self):
        return self._parameterized_variable_scopes

    @property
    def name(self):
        return self._name

    @property
    def parameter(self):
        return self._parameter_scope.parameter

    @property
    def parameterized_variables(self):
        return self._parameter_scope.parameterized_variables

    @property
    def parameter_size(self):
        return self.template_variable_scope.size

    @property
    def template_variables(self):
        return self.template_variable_scope.variables

    def __len__(self):
        return len(self.parameterized_variable_scopes)


class TemplateVariable(object):
    """Record necessary information about a template variable for reconstruction.
    """

    def __init__(self, variable, index, template_name):
        self._variable = variable
        self._index = index
        self._template_name = template_name

    @property
    def template_name(self):
        return self._template_name

    @property
    def variable(self):
        return self._variable

    @property
    def full_name(self):
        return self.variable.name

    @property
    def name(self):
        return self.variable.name.split(self.template_name)[-1][1:-2]

    @property
    def shape(self):
        return self.variable.get_shape().as_list()

    @property
    def size(self):
        if len(self.shape) != 0:
            return functools.reduce(operator.mul, self.shape)

        # Single scaler variable
        return 1

    @property
    def index(self):
        return self._index

    @property
    def dtype(self):
        return self.variable.dtype


class TemplateVariableScope(object):
    """Wrapped around the default variable scope to collect variable information.
    The template information will be used for the parameterization of model.
    """

    def __init__(self, name, dtype=tf.float32, exit_callback=None, *args, **kwargs):
        self._variable_scope = tf.variable_scope(name, *args, **kwargs)
        self._variables = None
        self._name = name
        if not dtype.is_floating:
            raise ValueError('Expected floating point type, got %s.' % dtype)
        self._dtype = dtype
        self._size = 0
        self._exit_callback = exit_callback

    def __enter__(self):
        return self._variable_scope.__enter__()

    def __exit__(self, etype, value, tb):
        retval = self._variable_scope.__exit__(etype, value, tb)

        # Collect variable information
        self._variables = []
        collection = tf.get_collection(
            key=tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.name
        )
        for v in collection:
            if v.dtype.base_dtype != self._dtype:
                # We do not parameterize non-float variables.
                continue

            d = TemplateVariable(v, self.size, self.name)
            self._size += d.size
            self._variables.append(d)

        if self._exit_callback is not None:
            self._exit_callback()

        return retval

    @property
    def name(self):
        return self._name

    @property
    def variables(self):
        """A list of dictionaries describing variables defined in this scope."""
        return self._variables

    @property
    def variable_scope(self):
        """The wrapped variable scope object."""
        return self._variable_scope

    @property
    def size(self):
        """The total size of variables."""
        return self._size

    @property
    def dtype(self):
        """The paramterized data type."""
        return self._dtype


class ParameterizedVariableScope(object):
    """Parameterized variable scope from template.

    Create a variable scope with custom variable getter to get variable from the
    single parameter vector.
    """

    def __init__(self, template, name, parameter=None, *args, **kwargs):
        """Construct the parameterized variable scope.

        Args:
            template: a TemplateVariableScope instance.
            name: a string containing the name of this variable scope.
        """
        # Make sure the template has already been created
        assert template.variables is not None
        
        self._template = template
        self._name = name
        self._variable_scope = tf.variable_scope(
            name, custom_getter=self._parameterized_variable_getter,
            reuse=tf.AUTO_REUSE, *args, **kwargs
        )

        with self._variable_scope:
            if parameter is not None:
                self._parameter = parameter
            else:
                flattened_variables = []
                for v in self._template.variables:
                    flattened_variables.append(
                        tf.reshape(v.variable.initialized_value(), [-1])
                    )
                self._parameter = tf.Variable(
                    tf.concat(flattened_variables, axis=0), dtype=template.dtype
                )

            self._parameterized_variables = None

    def _parameterized_variable_getter(self, getter, name, *args, **kwargs):
        """Custom variable getter.

        We need to return the corresponding parameterized tensor instead of an
        actual variable. This method should not be called directly.

        Args:
            getter: the true variable getter.
            name: name of the variable to get.
        """
        descoped_name = name.split(self.name)[-1][1:]
        if descoped_name in self.parameterized_variables:
            # We've found a trainable variable defined in the template
            return self.parameterized_variables[descoped_name]
        return getter(name, *args, **kwargs)

    def __enter__(self):
        return self._variable_scope.__enter__()

    def __exit__(self, etype, value, tb):
        return self._variable_scope.__exit__(etype, value, tb)

    @property
    def parameter(self):
        """The 1D parameter vector variable."""
        return self._parameter

    @property
    def parameterized_variables(self):
        """A dictionary mapping name to parameterized tensors(not variable)."""
        if self._parameterized_variables is None:
            # Only slice the variables if needed
            self._parameterized_variables = OrderedDict()
            for v in self._template.variables:
                self._parameterized_variables[v.name] = tf.reshape(
                    tf.slice(self._parameter, [v.index], [v.size]), v.shape
                )
        return self._parameterized_variables

    @property
    def variable_scope(self):
        """The wrapped variable scope object."""
        return self._variable_scope

    @property
    def template(self):
        """The corresponding template object."""
        return self._template

    @property
    def name(self):
        return self._name
