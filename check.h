/*
 * Copyright (c) 2014 Smilart and others.
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     Alexander Komarov alexander07k@gmail.com - implementation.
 */

#ifndef CHECK_H_
#define CHECK_H_

#include <string>
#include <vector>
#include <limits>
#include <time.h>

inline void throwString(std::string fileName, std::string lineNumber, std::string failedExpression) {
	throw "Check failed: " + failedExpression + ", file " + fileName + ", line " + lineNumber;
}

inline void throwString(std::string fileName, std::string lineNumber, std::string failedExpression, std::string comment) {
	throw "Check failed: " + failedExpression + " (" + comment + "), file " + fileName + ", line " + lineNumber;
}

#ifdef QUOTE
#error QUOTE macro defined not only in check.h
#endif

#ifdef QUOTE_VALUE
#error QUOTE_VALUE macro defined not only in check.h
#endif

#ifdef check
#error check macro defined not only in check.h
#endif

#define QUOTE(x) #x

#define QUOTE_VALUE(x) QUOTE(x)

#define check(expression, ...) \
{ \
	if (!(expression)) { \
		throwString(__FILE__, QUOTE_VALUE(__LINE__), #expression, ##__VA_ARGS__); \
	} \
}

#endif // CHECK_H_
