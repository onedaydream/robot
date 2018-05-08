#!/usr/bin/env python3
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a recognizer using the Google Assistant Library.

The Google Assistant Library has direct access to the audio API, so this Python
code doesn't need to record audio. Hot word detection "OK, Google" is supported.

It is available for Raspberry Pi 2/3 only; Pi Zero is not supported.
"""

import logging
import platform
import subprocess
import sys
#------------
import serial
import time
#------------

import aiy.assistant.auth_helpers
from aiy.assistant.library import Assistant
import aiy.audio
import aiy.voicehat
from google.assistant.library.event import EventType

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
)

#--------------------------------------------
def auto_mode():
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=0.1)
    arduino.flush()
    var = str('1')
    arduino.write(var.encode())

def hand_mode():
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=0.1)
    arduino.flush()
    var = str('2')
    arduino.write(var.encode())

def up():
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=0.1)
    arduino.flush()
    var = str('3')
    arduino.write(var.encode())

def down():
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=0.1)
    arduino.flush()
    var = str('4')
    arduino.write(var.encode())

def left():
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=0.1)
    arduino.flush()
    var = str('5')
    arduino.write(var.encode())

def right():
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=0.1)
    arduino.flush()
    var = str('6')
    arduino.write(var.encode())
#--------------------------------------------

def process_event(assistant, event):
    status_ui = aiy.voicehat.get_status_ui()
    if event.type == EventType.ON_START_FINISHED:
        status_ui.status('ready')
        if sys.stdout.isatty():
            print('Say "OK, Google" then speak, or press Ctrl+C to quit...')

    elif event.type == EventType.ON_CONVERSATION_TURN_STARTED:
        status_ui.status('listening')

    elif event.type == EventType.ON_RECOGNIZING_SPEECH_FINISHED and event.args:
        print('You said:', event.args['text'])
        text = event.args['text'].lower()
        if text == 'auto mode':
            assistant.stop_conversation()
            auto_mode()
        elif text == 'hand mode':
            assistant.stop_conversation()
            hand_mode()
        elif text == 'moving forward':
            assistant.stop_conversation()
            up()
        elif text == 'moving backward':
            assistant.stop_conversation()
            down()
        elif text == 'turn left':
            assistant.stop_conversation()
            left()
        elif text == 'turn right':
            assistant.stop_conversation()
            right()

    elif event.type == EventType.ON_END_OF_UTTERANCE:
        status_ui.status('thinking')

    elif (event.type == EventType.ON_CONVERSATION_TURN_FINISHED
          or event.type == EventType.ON_CONVERSATION_TURN_TIMEOUT
          or event.type == EventType.ON_NO_RESPONSE):
        status_ui.status('ready')

    elif event.type == EventType.ON_ASSISTANT_ERROR and event.args and event.args['is_fatal']:
        sys.exit(1)


def main():
    if platform.machine() == 'armv6l':
        print('Cannot run hotword demo on Pi Zero!')
        exit(-1)

    credentials = aiy.assistant.auth_helpers.get_assistant_credentials()
    with Assistant(credentials) as assistant:
        for event in assistant.start():
            process_event(assistant, event)


if __name__ == '__main__':
    main()
