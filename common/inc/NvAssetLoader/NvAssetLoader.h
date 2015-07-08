//----------------------------------------------------------------------------------
// File:        NvAssetLoader/NvAssetLoader.h
// SDK Version: v2.11 
// Email:       gameworks@nvidia.com
// Site:        http://developer.nvidia.com/
//
// Copyright (c) 2014-2015, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//----------------------------------------------------------------------------------

#ifndef NV_ASSET_LOADER_H
#define NV_ASSET_LOADER_H

#include <NvFoundation.h>

/// \file
/// Cross-platform binary asset loader.
/// Cross-platform binary asset loader.  Requires files
/// to be located in a subdirectory of the application's source
/// tree named "assets" in order to be able to find them.  This
/// is enforced so that assets will be automatically packed into
/// the application's APK on Android (ANT defaults to packing the
/// tree under "assets" into the binary assets of the APK).
///
/// On platforms that use file trees for storage (Windows and
/// Linux), the search method for finding each file passed in as the
/// partial path <filepath> is as follows:
/// - Start at the application's current working directory
/// - Do up to 10 times:
///     -# For each search path <search> in the search list:
///          -# Try to open <currentdir>/<search>/<filepath>
///          -# If it is found, return it
///          -# Otherwise, move to next path in <search> and iterate
///     -# Change directory up one level and iterate
///
/// On Android, the file opened is always <filepath>, since the "assets"
/// directory is known (it is the APK's assets).


/// Initializes the loader at application start.
/// Initializes the loader.  In most cases, the platform-specific
/// application framework or main loop should make this call.  It
/// requires a different argument on each platform
/// param[in] platform a platform-specific context pointer used by
/// the implementation
/// - On Android, this should be the app's AssetManager instance
/// - On Windows and Linux, this is currently ignored and should be NULL
///
/// \return true on success and false on failure
bool NvAssetLoaderInit(void* platform);

/// Shuts down the system
/// \return true on success and false on failure
bool NvAssetLoaderShutdown();

/// Adds a search path for finding the root of the assets tree.
/// Adds a search path to be prepended to "assets" when searching
/// for the correct assets tree.  Note that this must be a relative
/// path, and it is not used directly to find the file.  It is only
/// used on path-based platforms (Linux and Windows) to find the
/// "assets" directory.
/// \param[in] The relative path to add to the set of paths used to
/// find the "assets" tree.  See the package description for the
/// file search methods
/// \return true on success and false on failure
bool NvAssetLoaderAddSearchPath(const char *path);

/// Removes a search path from the lists.
/// \param[in] the path to remove
/// \return true on success and false on failure (not finding the path
/// on the list is considered success)
bool NvAssetLoaderRemoveSearchPath(const char *path);

/// Reads an asset file as a block.
/// Reads an asset file, returning a pointer to a block of memory
/// that contains the entire file, along with the length.  The block
/// is null-terminated for safety
/// \param[in] filePath the partial path (below "assets") to the file
/// \param[out] length the length of the file in bytes
/// \return a pointer to a null-terminated block containing the contents
/// of the file or NULL on error.  This block should be freed with a call
/// to #NvAssetLoaderFree when no longer needed.  Do NOT delete the block
/// with free or delete[]
char *NvAssetLoaderRead(const char *filePath, int32_t &length);

/// Frees a block returned from #NvAssetLoaderRead.
/// \param[in] asset a pointer returned from #NvAssetLoaderRead
/// \return true on success and false on failure
bool NvAssetLoaderFree(char* asset);


#endif
