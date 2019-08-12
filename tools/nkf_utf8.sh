#!/bin/bash

FILES=$1

if [ -z "$FILES" ]; then
	echo $0" filename"
	exit 1
fi

nkf -w --overwrite $FILES

