# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: translation_service.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='translation_service.proto',
  package='translate',
  syntax='proto3',
  serialized_options=b'\n\032sg.edu.nyp.signquest.protoB\027TranslationServiceProtoP\001',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x19translation_service.proto\x12\ttranslate\"\x1e\n\x0cImageRequest\x12\x0e\n\x06pixels\x18\x01 \x01(\x0c\"\x1f\n\x0fTranslatedReply\x12\x0c\n\x04\x63har\x18\x01 \x01(\t2X\n\x12TranslationService\x12\x42\n\tTranslate\x12\x17.translate.ImageRequest\x1a\x1a.translate.TranslatedReply\"\x00\x42\x37\n\x1asg.edu.nyp.signquest.protoB\x17TranslationServiceProtoP\x01\x62\x06proto3'
)




_IMAGEREQUEST = _descriptor.Descriptor(
  name='ImageRequest',
  full_name='translate.ImageRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='pixels', full_name='translate.ImageRequest.pixels', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=40,
  serialized_end=70,
)


_TRANSLATEDREPLY = _descriptor.Descriptor(
  name='TranslatedReply',
  full_name='translate.TranslatedReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='char', full_name='translate.TranslatedReply.char', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=72,
  serialized_end=103,
)

DESCRIPTOR.message_types_by_name['ImageRequest'] = _IMAGEREQUEST
DESCRIPTOR.message_types_by_name['TranslatedReply'] = _TRANSLATEDREPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ImageRequest = _reflection.GeneratedProtocolMessageType('ImageRequest', (_message.Message,), {
  'DESCRIPTOR' : _IMAGEREQUEST,
  '__module__' : 'translation_service_pb2'
  # @@protoc_insertion_point(class_scope:translate.ImageRequest)
  })
_sym_db.RegisterMessage(ImageRequest)

TranslatedReply = _reflection.GeneratedProtocolMessageType('TranslatedReply', (_message.Message,), {
  'DESCRIPTOR' : _TRANSLATEDREPLY,
  '__module__' : 'translation_service_pb2'
  # @@protoc_insertion_point(class_scope:translate.TranslatedReply)
  })
_sym_db.RegisterMessage(TranslatedReply)


DESCRIPTOR._options = None

_TRANSLATIONSERVICE = _descriptor.ServiceDescriptor(
  name='TranslationService',
  full_name='translate.TranslationService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=105,
  serialized_end=193,
  methods=[
  _descriptor.MethodDescriptor(
    name='Translate',
    full_name='translate.TranslationService.Translate',
    index=0,
    containing_service=None,
    input_type=_IMAGEREQUEST,
    output_type=_TRANSLATEDREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_TRANSLATIONSERVICE)

DESCRIPTOR.services_by_name['TranslationService'] = _TRANSLATIONSERVICE

# @@protoc_insertion_point(module_scope)
