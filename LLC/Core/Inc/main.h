/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32h7xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

void HAL_TIM_MspPostInit(TIM_HandleTypeDef *htim);

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define B1_Pin GPIO_PIN_13
#define B1_GPIO_Port GPIOC
#define SAFETY_SW_Pin GPIO_PIN_3
#define SAFETY_SW_GPIO_Port GPIOF
#define BRAKE_FEEDBACK_Pin GPIO_PIN_4
#define BRAKE_FEEDBACK_GPIO_Port GPIOF
#define CL57T_STEP_Pin GPIO_PIN_7
#define CL57T_STEP_GPIO_Port GPIOF
#define CL57T_ALARM_Pin GPIO_PIN_2
#define CL57T_ALARM_GPIO_Port GPIOC
#define LD1_Pin GPIO_PIN_0
#define LD1_GPIO_Port GPIOB
#define AUTO_MANUAL_Pin GPIO_PIN_9
#define AUTO_MANUAL_GPIO_Port GPIOE
#define STATUS_LED_Pin GPIO_PIN_12
#define STATUS_LED_GPIO_Port GPIOB
#define KEYA_HUSSAM_SEL_Pin GPIO_PIN_15
#define KEYA_HUSSAM_SEL_GPIO_Port GPIOD
#define RC_CH3_Pin GPIO_PIN_3
#define RC_CH3_GPIO_Port GPIOG
#define RPM_SENSOR_Pin GPIO_PIN_5
#define RPM_SENSOR_GPIO_Port GPIOG
#define CL57T_DIR_Pin GPIO_PIN_13
#define CL57T_DIR_GPIO_Port GPIOG

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
