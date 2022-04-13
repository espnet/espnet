#!/usr/bin/env python
# -*- coding: utf-8 -*-

from wikipedia_accent_parser import WikipediaAccentParser

accent_parser = WikipediaAccentParser()

assert accent_parser.retrieve_accent(u"Алексей") == u"алексе́й"

assert accent_parser.retrieve_accent(u"Андрей") == u"андре́й"

assert accent_parser.retrieve_accent(u"Васильев") == u"васи́льев"

print("SUCCESS")