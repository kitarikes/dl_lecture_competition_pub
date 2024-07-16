#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <FILE_ID> <OUTPUT_FILE_NAME>"
        exit 1
	fi

	FILE_ID=$1
	FILE_NAME=${2:-downloaded_file}

	# Create a cookie file
	COOKIE_FILE=$(mktemp /tmp/cookie.XXXXXX)

	# Initial request to get the warning code
	wget --quiet --save-cookies $COOKIE_FILE --keep-session-cookies \
		"https://drive.google.com/uc?export=download&id=${FILE_ID}" -O- \
		          | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > /tmp/confirm_code

	CONFIRM_CODE=$(< /tmp/confirm_code)

	# Use the confirm code to download the file
	wget --load-cookies $COOKIE_FILE \
		     "https://drive.google.com/uc?export=download&confirm=${CONFIRM_CODE}&id=${FILE_ID}"  \
			     -O ${FILE_NAME}
 
	# Cleanup
	rm $COOKIE_FILE /tmp/confirm_code
	
