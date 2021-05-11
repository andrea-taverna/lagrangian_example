import re
import logging
from typing import Callable, TextIO, Dict, Any

logger = logging.Logger(__name__)
logger.setLevel(logging.DEBUG)

RawData = Dict[str, Any]

TokenParser = Callable[[str, RawData, TextIO], RawData]

DEFAULT_KEYWORD_RE = "([a-zA-Z_]+)"
NUMBER_RE = "([-+]?\d+\.\d+|\d+)"
INTEGER_RE = "(\d+)"


def parse_num(keyword: str, parser) -> TokenParser:
    def _actual_parser(current_line: str, data: RawData, stream: TextIO) -> RawData:
        m = re.search(NUMBER_RE, current_line)
        return {keyword: parser(current_line[m.start() : m.end()])}

    return _actual_parser


def skip_keyword(current_line: str, data:RawData, stream: TextIO) -> RawData:
    return {}


def parse_stream_by_keywords(stream:TextIO, parsers_dict: Dict[str, TokenParser], keyword_re=DEFAULT_KEYWORD_RE) -> RawData:
    raw_data = {}

    for line_num, line in enumerate(stream):
        m = re.match(keyword_re, line)
        if m is None:
            logger.debug(f"No keyword found at line {line_num}")
        else:
            keyword = m.groups()[0]
            parser = parsers_dict.get(keyword, skip_keyword)
            logger.debug(f"Found keyword {keyword} at line {line_num}, calling function {parser}")
            raw_data.update(**parser(line, raw_data, stream))

    return raw_data