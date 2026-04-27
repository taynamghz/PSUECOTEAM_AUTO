/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
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
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define STEERING_ZERO_DEG   -32.0f
#define STEERING_LEFT_DEG   -57.0f  // -32 - 25 = -57
#define STEERING_RIGHT_DEG   -7.0f  // -32 + 25 = -7
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;

DAC_HandleTypeDef hdac1;

I2C_HandleTypeDef hi2c1;

UART_HandleTypeDef hlpuart1;
UART_HandleTypeDef huart3;

TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim3;
TIM_HandleTypeDef htim4;
TIM_HandleTypeDef htim17;

/* USER CODE BEGIN PV */

uint32_t startup_lockout_ms = 3000; // ignore RC for 3s after boot

volatile uint32_t rc_ch2_rising = 0;
volatile uint32_t rc_ch2_pulse = 0;
volatile uint8_t rc_ch2_ready = 0;

volatile uint32_t rc_ch1_rising = 0;
volatile uint32_t rc_ch1_pulse = 0;
volatile uint8_t rc_ch1_ready = 0;

volatile int32_t stepper_position = 0;
volatile int32_t stepper_target = 0;
volatile uint32_t stepper_delay_us = 500;

// Speed measurement
#define PULSES_PER_REV       4
#define WHEEL_CIRCUMFERENCE  1.52f
#define SPEED_TIMEOUT_MS     500

volatile uint32_t rpm_pulse_count = 0;
volatile uint32_t rpm_last_tick = 0;
volatile uint32_t rpm_last_interval_us = 0;
volatile uint32_t rpm_pulse_pending = 0;

float speed_kmh = 0.0f;
float speed_rpm = 0.0f;
float distance_km = 0.0f;
volatile uint32_t total_pulses = 0;

volatile uint32_t tim2_last_capture = 0;
volatile uint32_t tim2_interval_us = 0;
volatile uint8_t tim2_new_pulse = 0;

volatile uint8_t tim2_led_pulse = 0;

volatile uint8_t uart_tx_busy = 0;

// Jetson UART protocol
#define JETSON_START_BYTE   0xAA
#define JETSON_FRAME_LEN    5
#define CMD_IDLE            0x00
#define CMD_THROTTLE        0x01
#define CMD_BRAKE           0x02
#define CMD_STEER 0x03
#define JETSON_WATCHDOG_MS  200

uint8_t jetson_rx_buf[1];           // single byte DMA/interrupt receive
uint8_t jetson_frame[5];            // assembled frame
uint8_t jetson_frame_idx = 0;       // current byte index
volatile uint32_t jetson_last_valid = 0; // last valid frame timestamp
volatile uint8_t jetson_active = 0; // 1 = jetson control, 0 = manual/idle


uint16_t as5600_angle_raw = 0;
float as5600_angle_deg = 0.0f;

#define AS5600_ZERO_OFFSET 0

volatile uint8_t speed_tx_pending = 0;
uint8_t speed_tx_buf[5];

uint8_t led_effects_enabled = 0; // 1 = situational flashing, 0 = always steady ON

uint8_t clutch_state = 0;

float target_speed_kmh = 0.0f;
float speed_error_integral = 0.0f;
#define SPEED_KP 90.0f
#define SPEED_KI 180.0f
#define SPEED_MAX_KMH 30.0f

#define JETSON_STEER_MAX_DEG 25.0f
float smoothed_target_angle = 0.0f;
#define STEER_SMOOTH_RATE 25.0f
float actual_angle_deg = 0.0f;
float target_angle_deg = 0.0f;

float displayed_target_speed = 0.0f; // rate-limited version

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MPU_Config(void);
static void MX_GPIO_Init(void);
static void MX_DAC1_Init(void);
static void MX_I2C1_Init(void);
static void MX_LPUART1_UART_Init(void);
static void MX_TIM1_Init(void);
static void MX_TIM3_Init(void);
static void MX_TIM4_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_ADC1_Init(void);
static void MX_TIM17_Init(void);
static void MX_TIM2_Init(void);



/* USER CODE BEGIN PFP */
void update_speed(void);
uint16_t AS5600_ReadAngle(void);
uint8_t crc8_smbus(uint8_t *data, uint8_t len);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
#define DWT_DELAY_US(us) do { \
    uint32_t start = DWT->CYCCNT; \
    uint32_t cycles = (us) * (SystemCoreClock / 1000000); \
    while ((DWT->CYCCNT - start) < cycles); \
} while(0)
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MPU Configuration--------------------------------------------------------*/
  MPU_Config();

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DAC1_Init();
  MX_I2C1_Init();
  MX_LPUART1_UART_Init();
  MX_TIM1_Init();
  MX_TIM3_Init();
  MX_TIM4_Init();
  MX_USART3_UART_Init();
  MX_ADC1_Init();
  MX_TIM17_Init();
  MX_TIM2_Init();
  /* USER CODE BEGIN 2 */
//  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_3);
//  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 1500); // neutral arm
//  HAL_Delay(2000);
//  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);

  HAL_DAC_Start(&hdac1, DAC_CHANNEL_1);
  HAL_TIM_IC_Start_IT(&htim1, TIM_CHANNEL_1);
  HAL_TIM_IC_Start_IT(&htim1, TIM_CHANNEL_2);
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1);
  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 1833);
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_3);
  HAL_TIM_IC_Start_IT(&htim2, TIM_CHANNEL_1);
  HAL_TIM_Base_Start_IT(&htim17);  // stepper timer interrupt

  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

  rpm_last_tick = HAL_GetTick();

  HAL_UART_Receive_IT(&hlpuart1, jetson_rx_buf, 1);  // start UART receive interrupt

  // Safety startup delay — hold all outputs at zero for 2 seconds
  target_speed_kmh = 0.0f;
  speed_error_integral = 0.0f;
  jetson_active = 0;
  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);
  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 1833);
  HAL_Delay(2000);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

	  // Status LED — brake/throttle/idle indicator
	  // Status LED
	  static uint32_t led_blink_last = 0;

	  if (!led_effects_enabled)
	  {
	      // Always steady ON
	      HAL_GPIO_WritePin(STATUS_LED_GPIO_Port, STATUS_LED_Pin, GPIO_PIN_SET);
	  }
	  else if (!jetson_active)
	  {
	      // IDLE mode — slow flash 500ms
	      if (HAL_GetTick() - led_blink_last > 500)
	      {
	          led_blink_last = HAL_GetTick();
	          HAL_GPIO_TogglePin(STATUS_LED_GPIO_Port, STATUS_LED_Pin);
	      }
	  }
	  else if (__HAL_TIM_GET_COMPARE(&htim3, TIM_CHANNEL_1) > 1200)
	  {
	      // Brake active — flash 250ms
	      if (HAL_GetTick() - led_blink_last > 250)
	      {
	          led_blink_last = HAL_GetTick();
	          HAL_GPIO_TogglePin(STATUS_LED_GPIO_Port, STATUS_LED_Pin);
	      }
	  }
	  else if (speed_kmh > 0.5f)
	  {
	      // Running — solid ON
	      HAL_GPIO_WritePin(STATUS_LED_GPIO_Port, STATUS_LED_Pin, GPIO_PIN_SET);
	  }
	  else
	  {
	      // Full stop — solid ON
	      HAL_GPIO_WritePin(STATUS_LED_GPIO_Port, STATUS_LED_Pin, GPIO_PIN_SET);
	  }


	  // Send speed to Jetson + serial monitor every 100ms
	  static uint32_t last_speed_tx = 0;
	  if (HAL_GetTick() - last_speed_tx > 500)
	  {
	      last_speed_tx = HAL_GetTick();
	      printf("Target: %d.%02d | Actual: %d.%02d km/h | Angle: %d deg | SteerTarget: %d | tick:%lu\r\n",
	             (int)target_speed_kmh,
	             (int)((target_speed_kmh - (int)target_speed_kmh) * 100),
	             (int)speed_kmh,
	             (int)((speed_kmh - (int)speed_kmh) * 100),
	             (int)actual_angle_deg,
	             (int)target_angle_deg,
	             HAL_GetTick());

	      uint8_t speed_byte = (uint8_t)(speed_kmh * 10.0f);
	      uint8_t payload[3] = {0x02, 0x10, speed_byte};
	      uint8_t crc = crc8_smbus(payload, 3);
	      speed_tx_buf[0] = 0xBB;
	      speed_tx_buf[1] = 0x02;
	      speed_tx_buf[2] = 0x10;
	      speed_tx_buf[3] = speed_byte;
	      speed_tx_buf[4] = crc;
	      speed_tx_pending = 1;
	  }

	  if (speed_tx_pending && !uart_tx_busy)
	  {
	      speed_tx_pending = 0;
	      uart_tx_busy = 1;
	      HAL_UART_Transmit_IT(&hlpuart1, speed_tx_buf, 5);
	  }


	  // AS5600 angle read + steering PD controller
	  static uint32_t last_angle_read = 0;
	  if (HAL_GetTick() - last_angle_read > 10)
	  {
	      last_angle_read = HAL_GetTick();
	      as5600_angle_raw = AS5600_ReadAngle();
	      int16_t raw_signed = (int16_t)as5600_angle_raw;
	      if (raw_signed > 2048) raw_signed -= 4096;
	      actual_angle_deg = (raw_signed * 360.0f / 4096.0f) - STEERING_ZERO_DEG;

	      // Smooth target angle changes
	      if (target_angle_deg > smoothed_target_angle + STEER_SMOOTH_RATE)
	          smoothed_target_angle += STEER_SMOOTH_RATE;
	      else if (target_angle_deg < smoothed_target_angle - STEER_SMOOTH_RATE)
	          smoothed_target_angle -= STEER_SMOOTH_RATE;
	      else
	          smoothed_target_angle = target_angle_deg;

	      // Don't let smoothed target get more than 5° ahead of actual
	      float max_lead = 20.0f;
	      if (smoothed_target_angle > actual_angle_deg + max_lead)
	          smoothed_target_angle = actual_angle_deg + max_lead;
	      if (smoothed_target_angle < actual_angle_deg - max_lead)
	          smoothed_target_angle = actual_angle_deg - max_lead;

	      // Clamp to physical limits
	      float left_limit  = STEERING_LEFT_DEG  - STEERING_ZERO_DEG;
	      float right_limit = STEERING_RIGHT_DEG - STEERING_ZERO_DEG;
	      if (smoothed_target_angle < left_limit)  smoothed_target_angle = left_limit;
	      if (smoothed_target_angle > right_limit) smoothed_target_angle = right_limit;

	      if (jetson_active)
	      {
	          static float last_error = 0.0f;
	          float error = smoothed_target_angle - actual_angle_deg;
	          float derivative = error - last_error;
	          last_error = error;

	          float Kp = 15.0f;
	          float Kd = 0.0f;
	          float output = Kp * error + Kd * derivative;

	          // Clamp output to ±100 (like RC pulse range)
	          if (output > 100.0f)  output = 100.0f;
	          if (output < -100.0f) output = -100.0f;

	          float abs_output = output < 0 ? -output : output;

	          if (abs_output > 18.0f) // deadband
	          {
	              // Map output 2-100 → delay 200-30µs
	              uint32_t delay = (uint32_t)(300.0f - ((abs_output - 18.0f) / 98.0f) * 300.0f);
	              if (delay < 200)  delay = 200;
	              if (delay > 500) delay = 500;

	              int8_t new_direction = (output > 0) ? -1 : 1;

	              // Direction change — kick to overcome backlash
	              static int8_t last_direction = 0;
	              static uint32_t kick_start = 0;
	              static uint8_t kicking = 0;

	              if (new_direction != last_direction && last_direction != 0)
	              {
	                  kicking = 1;
	                  kick_start = HAL_GetTick();
	              }
	              last_direction = new_direction;

	              if (kicking)
	              {
	                  delay = 10; // full speed during kick *changed - was 30
	                  if (HAL_GetTick() - kick_start > 80) // kick lasts 80ms
	                      kicking = 0;
	              }

	              stepper_delay_us = delay;
	              stepper_target = new_direction;
	          }
	          else
	          {
	              stepper_target = 0;
	              stepper_delay_us = 200;
	          }
	      }
	  }



	  HAL_GPIO_WritePin(LD1_GPIO_Port, LD1_Pin,
	      actual_angle_deg > 0.0f ? GPIO_PIN_SET : GPIO_PIN_RESET);



	  update_speed();

	  // RC CH1 → stepper — encoder limited

	  if (!jetson_active && rc_ch1_ready && HAL_GetTick() > startup_lockout_ms)
	  {
	      rc_ch1_ready = 0;
	      uint32_t pulse = rc_ch1_pulse;

	      if (pulse >= 800 && pulse <= 2200)
	      {
	          if (pulse < 1000) pulse = 1000;
	          if (pulse > 2000) pulse = 2000;

	          if (pulse > 1550)
	          {
	              stepper_target = 1;
	              stepper_delay_us = 500 - ((pulse - 1550) * (500 - 50)) / 450;
	          }
	          else if (pulse < 1450)
	          {
	              stepper_target = -1;
	              stepper_delay_us = 500 - ((1450 - pulse) * (500 - 50)) / 450;
	          }
	          else
	          {
	              stepper_target = 0;
	              stepper_delay_us = 500;
	          }
	      }
	  }

	  if (!clutch_state)
	      stepper_target = 0;



	  // RC CH2 → Keya throttle + brake servo
	  if (rc_ch2_ready && HAL_GetTick() > startup_lockout_ms)
	  {
	      rc_ch2_ready = 0;
	      uint32_t pulse = rc_ch2_pulse;
	      uint32_t servo_ccr;

	      if (pulse >= 800 && pulse <= 2200)
	      {
	          if (pulse < 1450)
	          {
	              jetson_active = 0;
	              target_speed_kmh = 0.0f;
	              speed_error_integral = 0.0f;
	              displayed_target_speed = 0.0f;
	              __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);
	              if (pulse < 1000) pulse = 1000;
	              servo_ccr = 1833 - ((1450 - pulse) * (1833 - 500)) / 450;
	              static uint32_t last_servo_ccr = 500;
	              if (servo_ccr > last_servo_ccr + 20 || servo_ccr + 20 < last_servo_ccr)
	              {
	                  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, servo_ccr);
	                  last_servo_ccr = servo_ccr;
	              }
	          }
	          else if (!jetson_active)
	          {
	              if (pulse > 2000) pulse = 2000;
	              target_speed_kmh = pulse > 1500 ?
	                  ((pulse - 1500) * SPEED_MAX_KMH) / 500.0f : 0.0f;

	              servo_ccr = 500;
	              static uint32_t last_servo_ccr2 = 500;
	              if (servo_ccr > last_servo_ccr2 + 20 || servo_ccr + 20 < last_servo_ccr2)
	              {
	                  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, servo_ccr);
	                  last_servo_ccr2 = servo_ccr;
	              }
	          }
	      }
	  }
	  else
	  {
	      rc_ch2_ready = 0;
	      if (HAL_GetTick() <= startup_lockout_ms)
	      {
	          target_speed_kmh = 0.0f;
	          speed_error_integral = 0.0f;
	          displayed_target_speed = 0.0f;
	      }
	  }

	  // PI speed controller — runs every 20ms, outside RC block
	  static uint32_t last_pi_tick = 0;
	  if (HAL_GetTick() - last_pi_tick >= 20)
	  {
	      last_pi_tick = HAL_GetTick();

	      // Rate limit target
	      float max_step = 2.0f;
	      if (target_speed_kmh > displayed_target_speed + max_step)
	          displayed_target_speed += max_step;
	      else if (target_speed_kmh < displayed_target_speed - max_step)
	          displayed_target_speed -= max_step;
	      else
	          displayed_target_speed = target_speed_kmh;

	      float speed_error = displayed_target_speed - speed_kmh;

	      if (displayed_target_speed < 0.5f)
	      {
	          speed_error_integral = 0.0f;
	          __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);
	      }
	      else
	      {
	          speed_error_integral += speed_error * 0.02f;
	          if (speed_error_integral > 19999.0f) speed_error_integral = 19999.0f;
	          if (speed_error_integral < 0.0f) speed_error_integral = 0.0f;
	          float output = SPEED_KP * speed_error + SPEED_KI * speed_error_integral;
	          if (output > 19999.0f) output = 19999.0f;
	          if (output < 0.0f) output = 0.0f;
	          __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, (uint32_t)output);
	      }
	  }



	  // B1 button → EM clutch toggle
	  static uint8_t last_b1 = 0;
	  uint8_t b1 = HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_13);

	  if (b1 == GPIO_PIN_SET && last_b1 == GPIO_PIN_RESET)
	  {
	      clutch_state = !clutch_state;
	      HAL_GPIO_WritePin(GPIOF, GPIO_PIN_5, clutch_state ? GPIO_PIN_SET : GPIO_PIN_RESET);

	      // Release brake when clutch disengages
	      if (!clutch_state)
	      {
	          target_speed_kmh = 0.0f;
	          speed_error_integral = 0.0f;
	          __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);
	          __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 1833);
	      }
	  }
	  last_b1 = b1;

	  // Jetson watchdog **changed (added watchdog)
//	  if (jetson_active && (HAL_GetTick() - jetson_last_valid > JETSON_WATCHDOG_MS))
//	  {
//	      jetson_active = 0;
//	      target_speed_kmh = 0.0f;
//	      speed_error_integral = 0.0f;
//	      displayed_target_speed = 0.0f;
//	      target_angle_deg = 0.0f;
//	      smoothed_target_angle = 0.0f;
//	      stepper_target = 0;
//	      __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);
//	      __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 1833);
//	  }


  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = 64;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 12;
  RCC_OscInitStruct.PLL.PLLP = 1;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_3;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 4096;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ADC1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC1_Init(void)
{

  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_MultiModeTypeDef multimode = {0};
  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */

  /** Common config
  */
  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_ASYNC_DIV1;
  hadc1.Init.Resolution = ADC_RESOLUTION_16B;
  hadc1.Init.ScanConvMode = ADC_SCAN_DISABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  hadc1.Init.LowPowerAutoWait = DISABLE;
  hadc1.Init.ContinuousConvMode = ENABLE;
  hadc1.Init.NbrOfConversion = 1;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc1.Init.ConversionDataManagement = ADC_CONVERSIONDATA_DR;
  hadc1.Init.Overrun = ADC_OVR_DATA_PRESERVED;
  hadc1.Init.LeftBitShift = ADC_LEFTBITSHIFT_NONE;
  hadc1.Init.OversamplingMode = DISABLE;
  hadc1.Init.Oversampling.Ratio = 1;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure the ADC multi-mode
  */
  multimode.Mode = ADC_MODE_INDEPENDENT;
  if (HAL_ADCEx_MultiModeConfigChannel(&hadc1, &multimode) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_19;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC_SAMPLETIME_810CYCLES_5;
  sConfig.SingleDiff = ADC_SINGLE_ENDED;
  sConfig.OffsetNumber = ADC_OFFSET_NONE;
  sConfig.Offset = 0;
  sConfig.OffsetSignedSaturation = DISABLE;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */

}

/**
  * @brief DAC1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_DAC1_Init(void)
{

  /* USER CODE BEGIN DAC1_Init 0 */

  /* USER CODE END DAC1_Init 0 */

  DAC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN DAC1_Init 1 */

  /* USER CODE END DAC1_Init 1 */

  /** DAC Initialization
  */
  hdac1.Instance = DAC1;
  if (HAL_DAC_Init(&hdac1) != HAL_OK)
  {
    Error_Handler();
  }

  /** DAC channel OUT1 config
  */
  sConfig.DAC_SampleAndHold = DAC_SAMPLEANDHOLD_DISABLE;
  sConfig.DAC_Trigger = DAC_TRIGGER_NONE;
  sConfig.DAC_OutputBuffer = DAC_OUTPUTBUFFER_ENABLE;
  sConfig.DAC_ConnectOnChipPeripheral = DAC_CHIPCONNECT_DISABLE;
  sConfig.DAC_UserTrimming = DAC_TRIMMING_FACTORY;
  if (HAL_DAC_ConfigChannel(&hdac1, &sConfig, DAC_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN DAC1_Init 2 */

  /* USER CODE END DAC1_Init 2 */

}

/**
  * @brief I2C1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C1_Init(void)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  hi2c1.Init.Timing = 0x10C0ECFF;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Analogue filter
  */
  if (HAL_I2CEx_ConfigAnalogFilter(&hi2c1, I2C_ANALOGFILTER_ENABLE) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Digital filter
  */
  if (HAL_I2CEx_ConfigDigitalFilter(&hi2c1, 0) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

}

/**
  * @brief LPUART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_LPUART1_UART_Init(void)
{

  /* USER CODE BEGIN LPUART1_Init 0 */

  /* USER CODE END LPUART1_Init 0 */

  /* USER CODE BEGIN LPUART1_Init 1 */

  /* USER CODE END LPUART1_Init 1 */
  hlpuart1.Instance = LPUART1;
  hlpuart1.Init.BaudRate = 115200;
  hlpuart1.Init.WordLength = UART_WORDLENGTH_8B;
  hlpuart1.Init.StopBits = UART_STOPBITS_1;
  hlpuart1.Init.Parity = UART_PARITY_NONE;
  hlpuart1.Init.Mode = UART_MODE_TX_RX;
  hlpuart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  hlpuart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  hlpuart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  hlpuart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  hlpuart1.FifoMode = UART_FIFOMODE_DISABLE;
  if (HAL_UART_Init(&hlpuart1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&hlpuart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&hlpuart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&hlpuart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN LPUART1_Init 2 */

  /* USER CODE END LPUART1_Init 2 */

}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 115200;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  huart3.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart3.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart3.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart3, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart3, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_IC_InitTypeDef sConfigIC = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 199;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 65535;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
  if (HAL_TIM_IC_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterOutputTrigger2 = TIM_TRGO2_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigIC.ICPolarity = TIM_INPUTCHANNELPOLARITY_BOTHEDGE;
  sConfigIC.ICSelection = TIM_ICSELECTION_DIRECTTI;
  sConfigIC.ICPrescaler = TIM_ICPSC_DIV1;
  sConfigIC.ICFilter = 0;
  if (HAL_TIM_IC_ConfigChannel(&htim1, &sConfigIC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_IC_ConfigChannel(&htim1, &sConfigIC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_IC_InitTypeDef sConfigIC = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 199;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 4294967295;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
  if (HAL_TIM_IC_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigIC.ICPolarity = TIM_INPUTCHANNELPOLARITY_RISING;
  sConfigIC.ICSelection = TIM_ICSELECTION_DIRECTTI;
  sConfigIC.ICPrescaler = TIM_ICPSC_DIV1;
  sConfigIC.ICFilter = 0;
  if (HAL_TIM_IC_ConfigChannel(&htim2, &sConfigIC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */

}

/**
  * @brief TIM3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM3_Init(void)
{

  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 199;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 19999;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
  if (HAL_TIM_PWM_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 1500;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_LOW;
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */
  HAL_TIM_MspPostInit(&htim3);

}

/**
  * @brief TIM4 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM4_Init(void)
{

  /* USER CODE BEGIN TIM4_Init 0 */

  /* USER CODE END TIM4_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM4_Init 1 */

  /* USER CODE END TIM4_Init 1 */
  htim4.Instance = TIM4;
  htim4.Init.Prescaler = 199;
  htim4.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim4.Init.Period = 999;
  htim4.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim4.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
  if (HAL_TIM_PWM_Init(&htim4) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim4, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 500;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_LOW;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim4, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM4_Init 2 */

  /* USER CODE END TIM4_Init 2 */
  HAL_TIM_MspPostInit(&htim4);

}

/**
  * @brief TIM17 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM17_Init(void)
{

  /* USER CODE BEGIN TIM17_Init 0 */

  /* USER CODE END TIM17_Init 0 */

  /* USER CODE BEGIN TIM17_Init 1 */

  /* USER CODE END TIM17_Init 1 */
  htim17.Instance = TIM17;
  htim17.Init.Prescaler = 199;
  htim17.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim17.Init.Period = 999;
  htim17.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim17.Init.RepetitionCounter = 0;
  htim17.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
  if (HAL_TIM_Base_Init(&htim17) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM17_Init 2 */

  /* USER CODE END TIM17_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOF_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOF, GPIO_PIN_5|CL57T_STEP_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, LD1_Pin|STATUS_LED_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(CL57T_DIR_GPIO_Port, CL57T_DIR_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : SAFETY_SW_Pin BRAKE_FEEDBACK_Pin */
  GPIO_InitStruct.Pin = SAFETY_SW_Pin|BRAKE_FEEDBACK_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLDOWN;
  HAL_GPIO_Init(GPIOF, &GPIO_InitStruct);

  /*Configure GPIO pins : PF5 CL57T_STEP_Pin */
  GPIO_InitStruct.Pin = GPIO_PIN_5|CL57T_STEP_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOF, &GPIO_InitStruct);

  /*Configure GPIO pin : CL57T_ALARM_Pin */
  GPIO_InitStruct.Pin = CL57T_ALARM_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(CL57T_ALARM_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : LD1_Pin STATUS_LED_Pin */
  GPIO_InitStruct.Pin = LD1_Pin|STATUS_LED_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pin : AUTO_MANUAL_Pin */
  GPIO_InitStruct.Pin = AUTO_MANUAL_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLDOWN;
  HAL_GPIO_Init(AUTO_MANUAL_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : KEYA_HUSSAM_SEL_Pin */
  GPIO_InitStruct.Pin = KEYA_HUSSAM_SEL_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(KEYA_HUSSAM_SEL_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : RC_CH3_Pin */
  GPIO_InitStruct.Pin = RC_CH3_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLDOWN;
  HAL_GPIO_Init(RC_CH3_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : RPM_SENSOR_Pin */
  GPIO_InitStruct.Pin = RPM_SENSOR_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(RPM_SENSOR_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : CL57T_DIR_Pin */
  GPIO_InitStruct.Pin = CL57T_DIR_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(CL57T_DIR_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == LPUART1)
    {
        uart_tx_busy = 0;
        // Re-arm receive after transmit completes
        HAL_UART_Receive_IT(&hlpuart1, jetson_rx_buf, 1);
    }
}

//uint16_t AS5600_ReadAngle(void)
//{
//    uint8_t buf[2] = {0, 0};
//    uint8_t reg = 0x0E;
//    HAL_StatusTypeDef status;
//
//    status = HAL_I2C_Master_Transmit(&hi2c1, 0x36 << 1, &reg, 1, 5);
//    if (status != HAL_OK)
//    {
//        HAL_I2C_DeInit(&hi2c1);
//        HAL_Delay(1);
//        HAL_I2C_Init(&hi2c1);
//        return as5600_angle_raw; // return last known value
//    }
//
//    status = HAL_I2C_Master_Receive(&hi2c1, 0x36 << 1, buf, 2, 5);
//    if (status != HAL_OK)
//    {
//        HAL_I2C_DeInit(&hi2c1);
//        HAL_Delay(1);
//        HAL_I2C_Init(&hi2c1);
//        return as5600_angle_raw; // return last known value
//    }
//
//    return ((uint16_t)(buf[0] & 0x0F) << 8) | buf[1];
//} **changed

uint16_t AS5600_ReadAngle(void)
{
    static uint8_t fail_count = 0;
    uint8_t buf[2] = {0};
    uint8_t reg = 0x0E;

    if (HAL_I2C_Master_Transmit(&hi2c1, 0x36<<1, &reg, 1, 5) != HAL_OK ||
        HAL_I2C_Master_Receive(&hi2c1, 0x36<<1, buf, 2, 5) != HAL_OK)
    {
        fail_count++;
        if (fail_count > 5)
            stepper_target = 0;
        if (fail_count > 20)
        {
            fail_count = 0;
            HAL_I2C_DeInit(&hi2c1);
            HAL_Delay(1);
            HAL_I2C_Init(&hi2c1);
        }
        return as5600_angle_raw; // return last known on failure
    }

    fail_count = 0;
    uint16_t new_val = ((uint16_t)(buf[0] & 0x0F) << 8) | buf[1];

    // Reject raw=0 — safe for our steering range
    if (new_val == 0)
        return as5600_angle_raw;

    return new_val;
}


uint8_t crc8_smbus(uint8_t *data, uint8_t len)
{
    uint8_t crc = 0x00;
    for (uint8_t i = 0; i < len; i++)
    {
        crc ^= data[i];
        for (uint8_t b = 0; b < 8; b++)
            crc = (crc & 0x80) ? (crc << 1) ^ 0x07 : (crc << 1);
    }
    return crc;
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == LPUART1)
    {
        uint8_t byte = jetson_rx_buf[0];

        if (byte == JETSON_START_BYTE && jetson_frame_idx != 0)
            jetson_frame_idx = 0;

        jetson_frame[jetson_frame_idx++] = byte;

        if (jetson_frame_idx == JETSON_FRAME_LEN)
        {
            jetson_frame_idx = 0;

            if (jetson_frame[0] != JETSON_START_BYTE)
            {
                HAL_UART_Receive_IT(&hlpuart1, jetson_rx_buf, 1);
                return;
            }

            uint8_t crc = crc8_smbus(&jetson_frame[1], 3);
            if (crc != jetson_frame[4])
            {
                HAL_UART_Receive_IT(&hlpuart1, jetson_rx_buf, 1);
                return;
            }

            uint8_t cmd  = jetson_frame[2];
            uint8_t data = jetson_frame[3];
            jetson_last_valid = HAL_GetTick();

            switch (cmd)
            {
                case CMD_IDLE:
                    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);
                    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 1833);
                    target_speed_kmh = 0.0f;
                    speed_error_integral = 0.0f;
                    displayed_target_speed = 0.0f;
                    jetson_active = 0;
                    break;

                case CMD_BRAKE:
                    jetson_active = 1;
                    if (data > 0)
                        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 500);
                    else
                        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 1833);
                    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);
                    target_speed_kmh = 0.0f;
                    speed_error_integral = 0.0f;
                    displayed_target_speed = 0.0f;

                    break;

                case CMD_THROTTLE:
                    jetson_active = 1;
                    target_speed_kmh = ((float)data / 255.0f) * SPEED_MAX_KMH;
                    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 1833);
                    break;

                case CMD_STEER:
                    {
                    	jetson_active = 1;
                        float jetson_angle = -((float)data - 127.5f) * (2.0f * JETSON_STEER_MAX_DEG) / 255.0f;
                        float left_limit  = STEERING_LEFT_DEG  - STEERING_ZERO_DEG;
                        float right_limit = STEERING_RIGHT_DEG - STEERING_ZERO_DEG;
                        if (jetson_angle < left_limit)  jetson_angle = left_limit;
                        if (jetson_angle > right_limit) jetson_angle = right_limit;
                        target_angle_deg = jetson_angle;
                    }
                    break;
            }

            //uint8_t ack = cmd; *changed ack removed
            //HAL_UART_Transmit(&hlpuart1, &ack, 1, 100);
        }

        HAL_UART_Receive_IT(&hlpuart1, jetson_rx_buf, 1);
    }
}


void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM17)
    {
        if (stepper_target == 1)
        {
            HAL_GPIO_WritePin(CL57T_DIR_GPIO_Port, CL57T_DIR_Pin, GPIO_PIN_SET);
            HAL_GPIO_WritePin(CL57T_STEP_GPIO_Port, CL57T_STEP_Pin, GPIO_PIN_SET);
            DWT_DELAY_US(3); // 3µs — meets CL57T 2.5µs minimum **changed
            HAL_GPIO_WritePin(CL57T_STEP_GPIO_Port, CL57T_STEP_Pin, GPIO_PIN_RESET);
            __HAL_TIM_SET_AUTORELOAD(&htim17, stepper_delay_us);
        }
        else if (stepper_target == -1)
        {
            HAL_GPIO_WritePin(CL57T_DIR_GPIO_Port, CL57T_DIR_Pin, GPIO_PIN_RESET);
            HAL_GPIO_WritePin(CL57T_STEP_GPIO_Port, CL57T_STEP_Pin, GPIO_PIN_SET);
            DWT_DELAY_US(3); // 3µs — meets CL57T 2.5µs minimum **changed
            HAL_GPIO_WritePin(CL57T_STEP_GPIO_Port, CL57T_STEP_Pin, GPIO_PIN_RESET);
            __HAL_TIM_SET_AUTORELOAD(&htim17, stepper_delay_us);
        }
        else
        {
            __HAL_TIM_SET_AUTORELOAD(&htim17, 9999);
        }
    }
}

void update_speed(void)
{
    uint32_t now = HAL_GetTick();

    if ((now - rpm_last_tick) > SPEED_TIMEOUT_MS)
    {
        speed_kmh *= 0.85f;
        if (speed_kmh < 0.1f) speed_kmh = 0.0f;
        speed_rpm = 0.0f;
        return;
    }

    if (!tim2_new_pulse) return;
    tim2_new_pulse = 0;

    float interval_s = tim2_interval_us * 1e-6f;
    if (interval_s <= 0.0001f) return;

    float pulse_hz = 1.0f / interval_s;
    float rev_hz = pulse_hz / PULSES_PER_REV;
    speed_rpm = rev_hz * 60.0f;
    float raw_speed = rev_hz * WHEEL_CIRCUMFERENCE * 3.6f;

    float alpha;
    if (raw_speed <= 1.5f) alpha = 0.10f;
    else if (raw_speed >= 40.0f) alpha = 0.85f;
    else alpha = 0.10f + (raw_speed - 1.5f) / (40.0f - 1.5f) * (0.85f - 0.10f);

    speed_kmh = alpha * raw_speed + (1.0f - alpha) * speed_kmh;
    distance_km = (total_pulses / (float)PULSES_PER_REV) * WHEEL_CIRCUMFERENCE / 1000.0f;
    rpm_last_tick = now;
}

void HAL_TIM_IC_CaptureCallback(TIM_HandleTypeDef *htim)
{
    // TIM2 CH1 — RPM sensor input capture
    if (htim->Instance == TIM2 && htim->Channel == HAL_TIM_ACTIVE_CHANNEL_1)
    {
        uint32_t capture = HAL_TIM_ReadCapturedValue(htim, TIM_CHANNEL_1);
        uint32_t interval = capture - tim2_last_capture; // 32-bit no overflow needed
        tim2_last_capture = capture;
        if (interval > 30000)
        {
            tim2_interval_us = interval;
            tim2_new_pulse = 1;
            tim2_led_pulse = 1;
            total_pulses++;
            rpm_last_tick = HAL_GetTick(); // ← add this
        }


    }

    // TIM1 — RC input capture
    if (htim->Instance == TIM1)
    {
        // CH1 — RC steering
        if (htim->Channel == HAL_TIM_ACTIVE_CHANNEL_1)
        {
            static uint8_t ch1_state = 0;
            if (ch1_state == 0)
            {
                rc_ch1_rising = HAL_TIM_ReadCapturedValue(htim, TIM_CHANNEL_1);
                ch1_state = 1;
            }
            else
            {
                uint32_t falling = HAL_TIM_ReadCapturedValue(htim, TIM_CHANNEL_1);
                rc_ch1_pulse = (falling >= rc_ch1_rising) ? (falling - rc_ch1_rising) : (65535 - rc_ch1_rising + falling);
                rc_ch1_ready = 1;
                ch1_state = 0;
            }
        }

        // CH2 — RC throttle
        if (htim->Channel == HAL_TIM_ACTIVE_CHANNEL_2)
        {
            static uint8_t ch2_state = 0;
            if (ch2_state == 0)
            {
                rc_ch2_rising = HAL_TIM_ReadCapturedValue(htim, TIM_CHANNEL_2);
                ch2_state = 1;
            }
            else
            {
                uint32_t falling = HAL_TIM_ReadCapturedValue(htim, TIM_CHANNEL_2);
                rc_ch2_pulse = (falling >= rc_ch2_rising) ? (falling - rc_ch2_rising) : (65535 - rc_ch2_rising + falling);
                rc_ch2_ready = 1;
                ch2_state = 0;
            }
        }
    }
}

int _write(int file, char *ptr, int len)
{
    HAL_UART_Transmit(&huart3, (uint8_t*)ptr, len, 100);
    return len;
}


/* USER CODE END 4 */

 /* MPU Configuration */

void MPU_Config(void)
{
  MPU_Region_InitTypeDef MPU_InitStruct = {0};

  /* Disables the MPU */
  HAL_MPU_Disable();

  /** Initializes and configures the Region and the memory to be protected
  */
  MPU_InitStruct.Enable = MPU_REGION_ENABLE;
  MPU_InitStruct.Number = MPU_REGION_NUMBER0;
  MPU_InitStruct.BaseAddress = 0x0;
  MPU_InitStruct.Size = MPU_REGION_SIZE_4GB;
  MPU_InitStruct.SubRegionDisable = 0x87;
  MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
  MPU_InitStruct.AccessPermission = MPU_REGION_NO_ACCESS;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
  MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
  MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;

  HAL_MPU_ConfigRegion(&MPU_InitStruct);
  /* Enables the MPU */
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);

}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
