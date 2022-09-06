# Copyright 2022 Garena Online Private Limited.
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

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

project_path = osp.dirname(this_dir)
add_path(project_path)

add_path(osp.join(project_path, "libs", "criterions"))
add_path(osp.join(project_path, "libs", "datasets"))
add_path(osp.join(project_path, "libs", "encoders"))
add_path(osp.join(project_path, "libs", "evaluators"))
add_path(osp.join(project_path, "libs", "nerfheads"))
add_path(osp.join(project_path, "libs", "renders"))
add_path(osp.join(project_path, "libs", "trainers"))
add_path(osp.join(project_path, "libs", "masksegs"))
add_path(osp.join(project_path, "libs", "smpls"))
