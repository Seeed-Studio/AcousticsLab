#pragma once
#ifndef BOARD_CONFIG_H
#define BOARD_CONFIG_H

#if defined(PORTING_BOARD_MODEL_XIAO_S3) && PORTING_BOARD_MODEL_XIAO_S3
#include "xiao_s3.h"
#elif defined(PORTING_BOARD_MODEL_RESPEAKER_LITE) && PORTING_BOARD_MODEL_RESPEAKER_LITE
#include "respeaker_lite.h"
#elif defined(PORTING_BOARD_MODEL_RESPEAKER_XVF3800) && PORTING_BOARD_MODEL_RESPEAKER_XVF3800
#include "respeaker_xvf3800.h"
#else
#warning "No board model specified"
#endif

#endif // BOARD_CONFIG_H
