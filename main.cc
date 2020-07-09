/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ==============================================================================*/


#include <assert.h>
#include <fcntl.h>
#include <glob.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <set>
#include <vector>

#include <dirent.h>
//#include <sys/stat.h>
#include <fstream>
#include <string.h>


#include "wav_io.h"

class MemMappedFile {
 public:
  MemMappedFile(const std::string& filename) {
    const char* c_filename = filename.c_str();
    struct stat st;
    stat(c_filename, &st);
    filesize_ = st.st_size;
    fd_ = open(c_filename, O_RDONLY, 0);
    data_ = reinterpret_cast<uint8_t*>(
        mmap(NULL, filesize_, PROT_READ, MAP_PRIVATE, fd_, 0));
    assert(fd_ != -1);
    // Execute mmap
    if (data_ == MAP_FAILED) {
      fprintf(stderr, "mmap() failed with %p for '%s'\n", data_, filename.c_str());
    }
    assert(data_ != MAP_FAILED);
  }
  ~MemMappedFile() {
    int rc = munmap(data_, filesize_);
    assert(rc == 0);
    close(fd_);
  }

  size_t filesize_;
  int fd_;
  uint8_t* data_;
};

void TrimToLoudestSegment(const std::vector<float>& input,
                          int64_t desired_samples, std::vector<float>* output) {
  const int64_t input_size = input.size();
  if (desired_samples >= input_size) {
    *output = input;
    return;
  }

  float current_volume_sum = 0.0f;
  for (int64_t i = 0; i < desired_samples; ++i) {
    const float input_value = input[i];
    current_volume_sum += fabsf(input_value * input_value);
  }
  int64_t loudest_end_index = desired_samples;
  float loudest_volume = current_volume_sum;
  for (int64_t i = desired_samples; i < input_size; ++i) {
    const float trailing_value = input[i - desired_samples];
    current_volume_sum -= fabsf(trailing_value);
    const float leading_value = input[i];
    current_volume_sum += fabsf(leading_value);
    if (current_volume_sum > loudest_volume) {
      loudest_volume = current_volume_sum;
      loudest_end_index = i;
    }
  }
  const int64_t loudest_start_index = loudest_end_index - desired_samples;
  output->resize(desired_samples);
  std::copy(input.begin() + loudest_start_index,
            input.begin() + loudest_end_index, output->begin());
}

Status TrimFile(const std::string& input_filename,
                const std::string& output_filename,
                const int64_t desired_length_ms,
		const float min_volume) {
  MemMappedFile input_file(input_filename);

  std::vector<float> wav_samples;
  uint32_t sample_count;
  uint16_t channel_count;
  uint32_t sample_rate;
  Status load_wav_status = DecodeLin16WaveAsFloatVector(
      input_file.data_, input_file.filesize_, &wav_samples, &sample_count,
      &channel_count, &sample_rate);
  if (!load_wav_status.ok()) {
    std::cerr << "Failed to decode '" << input_filename
              << "' as a WAV: " << load_wav_status << std::endl;
    return load_wav_status;
  }

  // If we have a stereo or more recording, convert it down to mono.
  if (channel_count != 1) {
    const int mono_sample_count = sample_count / channel_count;
    std::vector<float> mono_samples(mono_sample_count);
    for (int i = 0; i < mono_sample_count; ++i) {
      const int frame_index = i * channel_count;
      float total = 0.0f;
      for (int c = 0; c < channel_count; ++c) {
    	total += wav_samples[frame_index + c];
      }
      mono_samples[i] = total / channel_count;
    }
    wav_samples = mono_samples;
  }

  const int64_t desired_samples = (desired_length_ms * sample_rate) / 1000;
  std::vector<float> trimmed_samples;
  TrimToLoudestSegment(wav_samples, desired_samples, &trimmed_samples);
  float total_volume = 0.0f;
  for (float trimmed_sample : trimmed_samples) {
    total_volume += fabsf(trimmed_sample);
  }
  const float average_volume = total_volume / desired_samples;
  if (average_volume < min_volume) {
    std::cerr << "Skipped '" << input_filename << "' as too quiet (" 
	      << average_volume << ")" << std::endl;
    return Status::OK();
  }

  std::string output_wav_data;
  Status save_wav_status =
      EncodeAudioAsS16LEWav(trimmed_samples.data(), sample_rate, 1,
                            trimmed_samples.size(), &output_wav_data);

  std::ofstream output_file(output_filename);
  output_file.write(output_wav_data.c_str(), output_wav_data.length());

  return Status::OK();
}

void SplitFilename(const std::string& full_path, std::string* dir,
                   std::string* filename) {
  std::size_t separator_index = full_path.find_last_of("/\\");
  *dir = full_path.substr(0, separator_index);
  *filename = full_path.substr(separator_index + 1);
}

bool isDirectory(const char* path)
{
  struct stat s;

  stat(path, &s);
  if (s.st_mode & S_IFDIR)
    return true;
  else
    return false;
}


void ParsePath(const char* path, std::vector<std::string>& filelist, std::vector<std::string>& txtlist)
{
  DIR* dir;
  struct dirent* ent;
  char real_path[PATH_MAX];

  if (realpath(path, real_path) == NULL) {
    std::cerr << "invalid path: " << path << std::endl;
    return;
  }

  if (!isDirectory(real_path)) {
    if (strstr(path, ".wav") != nullptr) {
      filelist.push_back(real_path);
    }
    if (strstr(path,".txt") != nullptr) {
	    txtlist.push_back(real_path);
    }
    return;
  }

  if ((dir = opendir(real_path)) != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir(dir)) != NULL) {
      if (!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."))
        continue;
      std::string full_path = real_path;
      full_path.append("/");
      full_path.append(ent->d_name);
      if (isDirectory(full_path.c_str()))
        ParsePath(full_path.c_str(), filelist, txtlist);
      else if (full_path.find(".wav") != std::string::npos)
        filelist.push_back(full_path);
      else if (full_path.find(".txt")!= std::string::npos)
	      txtlist.push_back(full_path);
    }
    closedir(dir);
  } else {
    /* could not open directory */
    perror("Could not open");
    return;
  }
}

void ParseDir(const char* path, std::set<std::string>& filelist)
{
  DIR* dir;
  struct dirent* ent;
  char real_path[PATH_MAX];

  if (realpath(path, real_path) == NULL) {
    std::cerr << "invalid path: " << path << std::endl;
    return;
  }
  if ((dir = opendir(real_path)) != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir(dir)) != NULL) {
      if (!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."))
        continue;
      std::string full_path = real_path;
      full_path.append("/");
      full_path.append(ent->d_name);
      if (isDirectory(full_path.c_str())) {
	filelist.insert(full_path);
        ParseDir(full_path.c_str(), filelist);
      }
    }
    closedir(dir);
  } else {
    /* could not open directory */
    perror("Could not open");
    return;
  }
}

std::string change_to_absolute_path(std::string p)
{
	char real_p[PATH_MAX];
	if (realpath(p.c_str(), real_p) == NULL) {
		std::cerr << "invalid path: " << p << std::endl;
		return "";
	}
	std::string s = real_p;
	return s;

}

int main(int argc, const char* argv[]) {
  if (argc < 3) {
    std::cerr
        << "You must supply paths to input and output wav files as arguments"
        << std::endl;
    return -1;
  }
  int l;
  float v;
  try{
    std::string length_ms = argv[3];
    l=std::stoi(length_ms);
  }
  catch(...){
    std::cout << "Something wrong in argv[3].." << std::endl;
    return -1;
  }
  try{
    std::string min_v = argv[4];
   v = std::stof(min_v);
  }
  catch(...){
    std::cout << "Something wrong in argv[4].." << std::endl;
    return -1;
  }
  // single wav file...
  const std::string input_glob = argv[1];
  const std::string output_root = argv[2];
  const std::string exte = ".wav";
  if ((input_glob.find(exte)!=std::string::npos) && (output_root.find(exte)!=std::string::npos)) {
	const int64_t desired_length_ms = (int64_t)l;//3840; //orig. 1000
    	const float min_volume = v;//0.000001f;
	Status trim_status =
	      TrimFile(input_glob, output_root, desired_length_ms, min_volume);
	if (!trim_status.ok()) {
	      std::cerr << "Failed on '" << input_glob << "' => '"
			<< output_root << "' with error " << trim_status;
	}
	return 0;
  }
  // directory as input...
  std::string real_in = change_to_absolute_path(input_glob);
  if (real_in.compare("")==0 ) {
	  std::cout << "Invalid path" << std::endl;
	  return -1;
  }
  std::string in_dir_name;
  std::string temp;
  SplitFilename(real_in.c_str(),&temp,&in_dir_name );
  //std::cout<<"absolute path of inp_dir  = "<<real_in<<" inp_dir name = "<< in_dir_name<<std::endl;
  std::string in_dir_name_slash = in_dir_name + "/";
  std::vector<std::string> input_filenames;
  std::vector<std::string> input_txt;
  ParsePath(real_in.c_str(), input_filenames, input_txt);
  mkdir(output_root.c_str(), ACCESSPERMS);
  std::string real_out = change_to_absolute_path(output_root);
  if (real_out.compare("")==0) {
	  std::cout << "Invalid path" << std::endl;
	  return -1;
  }
 //std::cout<<"absolute path of out_dir = "<<real_out<<std::endl;
 std::set<std::string> output_dirs;
 ParseDir(real_in.c_str(), output_dirs);
  std::vector<std::string> output_filenames;
  int wav_count = 0;
  for (const std::string& input_filename : input_filenames) {
    if (input_filename.find(".wav") != std::string::npos) {
	    wav_count+=1;
    }
    size_t pos = input_filename.find(in_dir_name_slash);
    std::string output_filename = real_out +"/"+ input_filename.substr(pos+in_dir_name_slash.length(), input_filename.length() - (pos+in_dir_name_slash.length()));
    output_filenames.push_back(output_filename);
  }

  for (const std::string& output_dir : output_dirs) {
    size_t pos = output_dir.find(in_dir_name_slash);
    std::string d_name = real_out + "/" + output_dir.substr(pos+in_dir_name_slash.length(), output_dir.length()-(pos+in_dir_name_slash.length()));
    mkdir(d_name.c_str(), ACCESSPERMS);
  }
  assert(input_filenames.size() == output_filenames.size());
  for (int64_t i = 0; i < input_filenames.size(); ++i) {
    const std::string input_filename = input_filenames[i];
    const std::string output_filename = output_filenames[i];
    const int64_t desired_length_ms = (int64_t)l;//3840; //orig. 1000
    const float min_volume = v;//0.000001f;
    Status trim_status =
      TrimFile(input_filename, output_filename, desired_length_ms, min_volume);
    if (!trim_status.ok()) {
      std::cerr << "Failed on '" << input_filename << "' => '"
                << output_filename << "' with error " << trim_status;
    }
  }
  for (int64_t i=0; i< input_txt.size(); i++) {
    size_t pos = input_txt[i].find(in_dir_name_slash);
    std::string output_filename = real_out +"/"+ input_txt[i].substr(pos+in_dir_name_slash.length(), input_txt[i].length() - (pos+in_dir_name_slash.length()));
    std::ifstream  src(input_txt[i], std::ios::binary);
    std::ofstream  dst(output_filename,std::ios::binary);
    dst << src.rdbuf();
  }
  return 0;
}
