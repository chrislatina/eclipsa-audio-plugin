// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <memory>

#include "PbRingBuffer.h"
#include "processors/file_output/iamf_export_utils/IAMFFileReader.h"

class BackgroundBuffer {
 public:
  BackgroundBuffer(const unsigned paddingSeconds, IAMFFileReader& decoder);
  ~BackgroundBuffer();

  bool isReady();
  size_t availableSamples() const;
  // Mark EOF when the decode thread has hit the end-of-stream AND the ring buffer has
  // been fully drained. Distinguishes a real EOF from a cold-start /
  // or playback underrun: returning 0 samples without isEof() means the
  // decode thread just hasn't refilled yet, not that there's nothing left.
  bool isEof() const;
  size_t readSamples(juce::AudioBuffer<float>& out, const unsigned startSample,
                     const unsigned numSamples);
  void seek(const size_t newFrameIdx);

  // Request abort of any seek request in progress 
  // to prevent locking up the decoder
  void requestAbortSeek() { abortSeek_.store(true); }

 private:
  void notifyTask();
  void decodeTask();

  IAMFFileReader& decoder_;
  size_t padSamples_, absSamplePos_;
  juce::SpinLock bufferLock_;
  std::unique_ptr<PbRingBuffer> pbuffer_;
  // Decoding thread and control
  std::atomic_bool stop_, eof_;
  // Set externally via requestAbortSeek() parsing frame.
  // Cleared at the start of each seek() call so a fresh seek isn't stale
  std::atomic_bool abortSeek_{false};
  std::condition_variable cv_;
  std::mutex cvm_;
  std::thread decodeThread_;
};