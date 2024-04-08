# llm-zhipu

[![PyPI](https://img.shields.io/pypi/v/llm-zhipu.svg)](https://pypi.org/project/llm-zhipu/)
[![Changelog](https://img.shields.io/github/v/release/noahlias/llm-zhipu?include_prereleases&label=changelog)](https://github.com/noahlias/llm-zhipu/releases)
[![Tests](https://github.com/noahlias/llm-zhipu/actions/workflows/test.yml/badge.svg)](https://github.com/noahlias/llm-zhipu/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/noahlias/llm-zhipu/blob/main/LICENSE)

ChatGLM(_智浦清言_)


## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-zhipu
```
## Usage

Usage instructions go here.

![example](./assets/example.webp)

```
llm image-identify-chatglm -l 'https://img1.baidu.com/it/u=1369931113,3388870256&fm=253&app=138&size=w931&n=0&f=JPEG&fmt=auto?sec=1703696400&t=f3028c7a1dca43a080aeb8239f09cc2f'
```
![output](./assets/output.jpg)


## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-zhipu
python3 -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
llm install -e '.[test]'
```
To run the tests:
```bash
pytest
```
