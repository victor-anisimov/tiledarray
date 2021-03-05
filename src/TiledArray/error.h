/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef TILEDARRAY_ERROR_H__INCLUDED
#define TILEDARRAY_ERROR_H__INCLUDED

#include <TiledArray/config.h>
#include <exception>

namespace TiledArray {

class Exception : public std::exception {
 public:
  Exception(const char* m) : message_(m) {}

  virtual const char* what() const noexcept { return message_; }

 private:
  const char* message_;
};  // class Exception

/// Place a break point on this function to stop before TiledArray exceptions
/// are thrown.
inline void exception_break() {}
}  // namespace TiledArray

#define TA_STRINGIZE(s) #s

#define TA_EXCEPTION_MESSAGE(file, line, mess) \
  "TiledArray: exception at " file "(" TA_STRINGIZE(line) "): " mess

/// throws TiledArray::Exception with message \p m annotated with the file name
/// and line number
/// \param m a C-style string constant
#define TA_EXCEPTION(m)                                                       \
  do {                                                                        \
    TiledArray::exception_break();                                            \
    throw TiledArray::Exception(TA_EXCEPTION_MESSAGE(__FILE__, __LINE__, m)); \
  } while (0)

/// TiledArray assertion is configured to throw TiledArray::Exception
/// if \p a is false
/// \param a an expression convertible to bool
#define TA_ASSERT(a)                             \
  do {                                           \
    if (!(a)) TA_EXCEPTION("assertion failure"); \
  } while (0)

#ifdef TILEDARRAY_NO_USER_ERROR_MESSAGES
#define TA_USER_ERROR_MESSAGE(m)
#else
#include <iostream>
#define TA_USER_ERROR_MESSAGE(m) \
  std::cerr << "!! ERROR TiledArray: " << m << "\n";
#endif  // TILEDARRAY_NO_USER_ERROR_MESSAGES

#endif  // TILEDARRAY_ERROR_H__INCLUDED
