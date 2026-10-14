#include <Arduino.h>
#include <Wire.h>

// ==== Pin mapping ====
#define ENC_L_A PA6
#define ENC_L_B PA7
#define ENC_R_A PA1
#define ENC_R_B PA0
#define PWMA PA8
#define AIN1 PB10
#define AIN2 PB11
#define PWMB PA9
#define BIN1 PB12
#define BIN2 PB13
#define LED_PIN PC13

// ============================================================
//  IMUPLUS yaw zero
// ============================================================
#define STARTUP_MAP_HEADING_DEG 0.0f
#define GYRO_Z_SIGN             1.0f

// ============================================================
//  BNO055
// ============================================================
#define BNO_CHIP_ID_REG   0x00
#define BNO_CHIP_ID_VAL   0xA0
#define BNO_OPR_MODE      0x3D
#define BNO_PWR_MODE      0x3E
#define BNO_SYS_TRIGGER   0x3F
#define BNO_UNIT_SEL      0x3B
#define BNO_MODE_CONFIG   0x00
#define BNO_MODE_IMUPLUS  0x08

uint8_t BNO_ADDR = 0x28;
float headingZeroRaw = 0.0f;

// ============================================================
//  Trạng thái xe
//  MODE_RUN  : chạy tự động theo lộ trình trong loop
//  MODE_STOP : dừng hoàn toàn
// ============================================================
enum RobotMode { MODE_RUN, MODE_STOP };
volatile RobotMode robotMode = MODE_STOP;

// ============================================================
//  CẤU HÌNH ĐIỀU HƯỚNG XE (ĐI THẲNG, QUAY TRÁI, DỪNG)
// ============================================================
// Góc quay trái mỗi lần (độ)
const float TARGET_TURN_DEG = 90.0f; // Quay trái 90 độ

// Thời gian tạm dừng giữa các chuyển động (ms) để triệt tiêu quán tính
const unsigned long PAUSE_BETWEEN_STATES_MS = 400;

// Các trạng thái điều hướng trong loop
enum NavState {
    NAV_DRIVE_STRAIGHT,        // Đi thẳng giữ hướng (bù góc + bù lệch encoder)
    NAV_PAUSE_AFTER_STRAIGHT,  // Dừng ổn định trước khi quay
    NAV_TURN_LEFT,             // Quay trái 90 độ
    NAV_PAUSE_AFTER_TURN       // Dừng ổn định sau khi quay
};
volatile NavState navState = NAV_DRIVE_STRAIGHT;
float targetHeading = 0.0f;    // Hướng mục tiêu hiện tại (độ)

// Thông số quay góc tại chỗ
const int   TURN_PWM_MAX       = 75;    // Tốc độ quay tối đa
const int   TURN_PWM_MIN       = 48;    // Tốc độ quay tối thiểu thắng ma sát sàn
const float KP_TURN            = 1.5f;  // Hệ số P cho quay góc
const float TURN_TOLERANCE_DEG = 1.8f;  // Ngưỡng sai số góc coi như quay xong (độ)

// PWM hiện tại do Pi ghi đè (−255…255).  −999 = chưa có lệnh từ Pi
volatile int overridePwmL = -999;
volatile int overridePwmR = -999;

volatile long    encLeft    = 0;
volatile long    encRight   = 0;
volatile uint8_t lastLeftA  = 0;
volatile uint8_t lastRightA = 0;

const float leftPPR   = 755.0f;
const float rightPPR  = 755.0f;
const float wheelCirc = 2.0f * 3.1416f * 32.5f;
const float WHEEL_BASE_MM = 145.0f; // Khoảng cách giữa 2 bánh xe (mm)

// PWM đi thẳng mặc định (khi không có override)
const int BASE_PWM = 60;
// Khởi động mềm: tăng tốc từ từ tránh giật và trượt bánh
const float START_PWM_MIN = 15.0f;        // Bắt đầu từ 15 (rất êm, 2 bánh cùng lăn đều)
const float ACCEL_RAMP_MS = 1200.0f;      // Thời gian tăng tốc từ 15 lên BASE_PWM (1200ms)
volatile unsigned long straightStartMs = 0;

// Hệ số điều khiển đi thẳng
float KP_HEADING = 3.8f;
float KD_HEADING = 0.08f;               // TẦNG 4: Hệ số vi phân D dập tắt lắc lư góc
float K_ENC_DIFF = 0.20f;

// TẦNG 4: Thông số kháng nhiễu và làm êm cơ cấu chấp hành
const float HEADING_DEADZONE_DEG = 0.5f;   // Vùng chết góc lái chống hiện tượng lắc đuôi
const float V_MAX_WHEEL_MM_S     = 650.0f; // TẦNG 3: Vận tốc bánh xe tối đa vật lý (lọc xung gai encoder)
const int   MAX_PWM_SLEW_STEP    = 8;      // Giới hạn dốc tăng giảm PWM mỗi 20ms (chống giật, trượt bánh)

// Trạng thái vận tốc bánh xe và PWM mượt mà
float vL_filtered = 0.0f;
float vR_filtered = 0.0f;
int currentPwmL   = 0;
int currentPwmR   = 0;

// Chu kỳ điều khiển (ms)
const unsigned long STRAIGHT_INTERVAL_MS = 20;

// Biến điều khiển chu kỳ và encoder
unsigned long lastControlMs = 0;
long prevEncL = 0;
long prevEncR = 0;
unsigned long pauseStartMs = 0;

// Encoder mốc khi bắt đầu đi thẳng
volatile long startEncL = 0;
volatile long startEncR = 0;
volatile float encDiffOffset = 0.0f;

// Trim bù vận tốc bánh (điều khiển bù lệch trực tiếp qua bàn phím)
volatile int trimPwmL = 0;
volatile int trimPwmR = 0;

// Chu kỳ gửi $IMU lên Pi (ms)
const unsigned long IMU_INTERVAL_MS = 50;

unsigned long lastDriveMs = 0;

// ============================================================
//  BNO055 helpers
// ============================================================
bool bnoWrite(uint8_t reg, uint8_t val) {
    Wire.beginTransmission(BNO_ADDR);
    Wire.write(reg); Wire.write(val);
    return Wire.endTransmission(true) == 0;
}

bool bnoRead(uint8_t reg, uint8_t len, uint8_t *buf) {
    Wire.beginTransmission(BNO_ADDR);
    Wire.write(reg);
    if (Wire.endTransmission(true) != 0) return false;
    uint8_t got = Wire.requestFrom(BNO_ADDR, len);
    if (got < len) return false;
    for (uint8_t i = 0; i < len; i++) buf[i] = Wire.read();
    return true;
}

static inline int16_t to16(uint8_t lo, uint8_t hi) {
    return (int16_t)((uint16_t)(hi << 8) | lo);
}

static inline float wrap180(float d) {
    while (d >= 180.0f)  d -= 360.0f;
    while (d < -180.0f) d += 360.0f;
    return d;
}

// ============================================================
//  Bộ lọc Kalman 1D kết hợp Encoder (Yaw Rate) + BNO055 (Heading)
// ============================================================
struct KalmanFilter {
    float angle;      // Góc hướng ước lượng (deg)
    float bias;       // Trôi góc / bias vận tốc (deg/s)
    float P[2][2];    // Ma trận hiệp phương sai sai số
    float Q_angle;    // Nhiễu quá trình dự đoán (Encoder)
    float Q_bias;     // Nhiễu quá trình bias
    float R_measure;  // Nhiễu đo lường cảm biến (BNO055)
    float yawRate;    // TẦNG 2: Vận tốc góc thực tế ước lượng (deg/s)

    void init(float initAngle = 0.0f) {
        angle = initAngle;
        bias  = 0.0f;
        yawRate = 0.0f;
        P[0][0] = 1.0f; P[0][1] = 0.0f;
        P[1][0] = 0.0f; P[1][1] = 1.0f;
        Q_angle   = 0.001f;
        Q_bias    = 0.003f;
        R_measure = 0.05f;
    }

    float update(float newAngleBNO, float rateEnc, float dt) {
        // 1. Dự đoán trạng thái dựa trên tốc độ quay từ Encoder (đã qua lọc Tầng 3)
        float rate = rateEnc - bias;
        angle = wrap180(angle + dt * rate);

        // Dự đoán hiệp phương sai
        P[0][0] += dt * (dt * P[1][1] - P[0][1] - P[1][0] + Q_angle);
        P[0][1] -= dt * P[1][1];
        P[1][0] -= dt * P[1][1];
        P[1][1] += Q_bias * dt;

        // 2. Cập nhật đo lường từ BNO055
        float y = wrap180(newAngleBNO - angle);
        float S = P[0][0] + R_measure;
        float K0 = P[0][0] / S;
        float K1 = P[1][0] / S;

        angle = wrap180(angle + K0 * y);
        bias += K1 * y;
        yawRate = rate;

        float P00_temp = P[0][0];
        float P01_temp = P[0][1];

        P[0][0] -= K0 * P00_temp;
        P[0][1] -= K0 * P01_temp;
        P[1][0] -= K1 * P00_temp;
        P[1][1] -= K1 * P01_temp;

        return angle;
    }
};

KalmanFilter kfHeading;

float readHeadingRaw() {
    static float lastValidHeading = 0.0f;
    uint8_t e[2];
    // TẦNG 1: Fail-safe chống sập đọc I2C do nhiễu xung động cơ, giữ lại góc trước
    if (!bnoRead(0x1A, 2, e)) return lastValidHeading;
    float d = to16(e[0], e[1]) / 16.0f + 180.0f;
    while (d <    0.0f) d += 360.0f;
    while (d >= 360.0f) d -= 360.0f;
    lastValidHeading = d;
    return d;
}

// TẦNG 1: Lọc thông thấp (EMA) góc thô BNO055 để loại bỏ rung lắc cơ học
float readHeadingMap() {
    static float filteredHeadingMap = 0.0f;
    static bool initHeadingFilter   = false;
    float rawMap = wrap180(readHeadingRaw() - headingZeroRaw + STARTUP_MAP_HEADING_DEG);
    if (!initHeadingFilter) {
        filteredHeadingMap = rawMap;
        initHeadingFilter  = true;
    } else {
        float diff = wrap180(rawMap - filteredHeadingMap);
        filteredHeadingMap = wrap180(filteredHeadingMap + 0.70f * diff);
    }
    return filteredHeadingMap;
}

// ============================================================
//  Gửi gói $IMU lên Pi
//  Định dạng: $IMU,<v_mm_s>,<heading_deg>
//  heading chỉ để Pi đọc/log, STM32 không dùng để điều khiển
// ============================================================
void sendIMUPacket() {
    long cL, cR;
    noInterrupts(); cL = encLeft; cR = encRight; interrupts();

    static long prevL = 0, prevR = 0;
    static unsigned long prevMs = 0;
    unsigned long nowMs = millis();
    float dt = (nowMs - prevMs) / 1000.0f;
    if (dt < 1e-4f) dt = 1e-4f;

    float distL  = ((cL - prevL) / leftPPR)  * wheelCirc;
    float distR  = ((cR - prevR) / rightPPR) * wheelCirc;
    float v_mm_s = (distL + distR) / (2.0f * dt);
    prevL = cL; prevR = cR; prevMs = nowMs;

    Serial.print("$IMU,");
    Serial.print(v_mm_s, 2);
    Serial.print(",");
    Serial.println(readHeadingMap(), 2);
}

// ============================================================
//  Motor
// ============================================================
void setupPWM() {
    pinMode(AIN1,OUTPUT); pinMode(AIN2,OUTPUT); pinMode(PWMA,OUTPUT);
    pinMode(BIN1,OUTPUT); pinMode(BIN2,OUTPUT); pinMode(PWMB,OUTPUT);
    analogWrite(PWMA,0);  analogWrite(PWMB,0);
}

void setWheelPWM(int pwmLeft, int pwmRight) {
    if (pwmLeft  >= 0) { digitalWrite(AIN1,LOW);  digitalWrite(AIN2,HIGH); analogWrite(PWMA, pwmLeft);  }
    else               { digitalWrite(AIN1,HIGH); digitalWrite(AIN2,LOW);  analogWrite(PWMA,-pwmLeft);  }
    if (pwmRight >= 0) { digitalWrite(BIN1,LOW);  digitalWrite(BIN2,HIGH); analogWrite(PWMB, pwmRight); }
    else               { digitalWrite(BIN1,HIGH); digitalWrite(BIN2,LOW);  analogWrite(PWMB,-pwmRight); }
}

// ============================================================
//  BNO055 init
// ============================================================
bool bnoInit() {
    uint8_t candidates[] = {0x28, 0x29};
    bool found = false;
    for (int i = 0; i < 2; i++) {
        Wire.beginTransmission(candidates[i]);
        if (Wire.endTransmission(true) == 0) {
            BNO_ADDR = candidates[i]; found = true;
            Serial.print(">>> BNO055 tai: 0x"); Serial.println(BNO_ADDR, HEX); break;
        }
    }
    if (!found) { Serial.println("!!! KHONG TIM THAY BNO055!"); return false; }
    uint8_t chipId = 0;
    for (int i = 0; i < 10; i++) {
        if (bnoRead(BNO_CHIP_ID_REG, 1, &chipId)) break; delay(100);
    }
    if (chipId != BNO_CHIP_ID_VAL) return false;
    bnoWrite(BNO_OPR_MODE, BNO_MODE_CONFIG); delay(25);
    bnoWrite(BNO_PWR_MODE, 0x00);            delay(10);
    bnoWrite(BNO_UNIT_SEL, 0x00);            delay(10);
    bnoWrite(BNO_SYS_TRIGGER, 0x00);         delay(10);
    bnoWrite(BNO_OPR_MODE, BNO_MODE_IMUPLUS); delay(700);
    Serial.println(">>> BNO055 IMUPLUS OK");
    return true;
}

// ============================================================
//  Encoder ISR
// ============================================================
void isr_leftA() {
    uint8_t a = digitalRead(ENC_L_A), b = digitalRead(ENC_L_B);
    if (a != lastLeftA) { if (a == b) encLeft++; else encLeft--; lastLeftA = a; }
}
void isr_rightA() {
    uint8_t a = digitalRead(ENC_R_A), b = digitalRead(ENC_R_B);
    if (a != lastRightA) { if (a == b) encRight++; else encRight--; lastRightA = a; }
}

// ============================================================
//  Nhận lệnh từ Pi
//
//  CMD,START          → bắt đầu đi thẳng liên tục
//  CMD,STOP           → dừng xe hoàn toàn
//  CMD,IR,LEFT        → IR trái chạm line → quay phải (override PWM)
//  CMD,IR,RIGHT       → IR phải chạm line → quay trái (override PWM)
//  CMD,IR,CLEAR       → cả 2 IR về 0 → huỷ override, về BASE_PWM thẳng
//  CMD,DRIVE,<L>,<R>  → Pi ghi đè PWM 2 bánh trực tiếp (−255…255)
//  CMD,CLEAR          → huỷ override bất kỳ, về BASE_PWM thẳng
// ============================================================
#define CMD_BUF_SIZE 64
char    cmdBuf[CMD_BUF_SIZE];
uint8_t cmdBufIdx = 0;

void processPiCommand(const char* cmd) {
    char buf[CMD_BUF_SIZE];
    strncpy(buf, cmd, CMD_BUF_SIZE - 1);
    buf[CMD_BUF_SIZE - 1] = '\0';

    char* tok  = strtok(buf, ",");
    if (!tok || strcmp(tok, "CMD") != 0) return;
    char* name = strtok(NULL, ",");
    if (!name) return;

    if (strcmp(name, "START") == 0) {
        // 1. Dừng motor ngay lập tức
        setWheelPWM(0, 0);
        overridePwmL = -999;
        overridePwmR = -999;
        delay(150); // Chờ xe hoàn toàn đứng yên tĩnh trên sàn

        // 2. Hiệu chỉnh IMU: Lấy trung bình 5 mẫu để SET GÓC 0 ĐỘ CHÍNH XÁC TẠI ĐÂY
        float sumHeading = 0.0f;
        for (int i = 0; i < 5; i++) {
            sumHeading += readHeadingRaw();
            delay(20);
        }
        headingZeroRaw = sumHeading / 5.0f;
        targetHeading = 0.0f;

        // 3. Khởi tạo lại Kalman Filter với góc 0.0 độ
        kfHeading.init(0.0f);

        // 4. Reset mốc Encoder
        noInterrupts();
        startEncL = encLeft;
        startEncR = encRight;
        prevEncL  = encLeft;
        prevEncR  = encRight;
        interrupts();

        // 5. Đồng bộ thời gian điều khiển để không bị vọt dt
        lastControlMs   = millis();
        straightStartMs = millis();

        // 6. Chuyển trạng thái đi thẳng
        trimPwmL = 0;
        trimPwmR = 0;
        encDiffOffset = 0.0f;
        vL_filtered = 0.0f;
        vR_filtered = 0.0f;
        currentPwmL = 0;
        currentPwmR = 0;
        navState  = NAV_DRIVE_STRAIGHT;
        robotMode = MODE_RUN;
        Serial.println(">>> CMD,START: Bat dau di thang!");
        return;
    }

    if (strcmp(name, "STOP") == 0) {
        robotMode = MODE_STOP;
        setWheelPWM(0, 0);
        trimPwmL = 0;
        trimPwmR = 0;
        encDiffOffset = 0.0f;
        currentPwmL = 0;
        currentPwmR = 0;
        vL_filtered = 0.0f;
        vR_filtered = 0.0f;
        Serial.println(">>> CMD,STOP: Dung xe!");
        return;
    }

    if (strcmp(name, "RESUME") == 0 || strcmp(name, "RUN") == 0) {
        if (robotMode == MODE_STOP) {
            noInterrupts();
            startEncL = encLeft;
            startEncR = encRight;
            prevEncL  = encLeft;
            prevEncR  = encRight;
            interrupts();
            encDiffOffset = 0.0f;
            trimPwmL = 0;
            trimPwmR = 0;
            vL_filtered = 0.0f;
            vR_filtered = 0.0f;
            currentPwmL = 0;
            currentPwmR = 0;
            straightStartMs = millis();
            lastControlMs   = millis();
            navState  = NAV_DRIVE_STRAIGHT;
            robotMode = MODE_RUN;
            Serial.println(">>> CMD,RESUME: Tiep tuc di thang!");
        }
        return;
    }

    if (strcmp(name, "TURN") == 0 || strcmp(name, "TURN_LEFT") == 0) {
        if (navState == NAV_DRIVE_STRAIGHT) {
            setWheelPWM(0, 0);
            trimPwmL = 0;
            trimPwmR = 0;
            encDiffOffset = 0.0f;
            currentPwmL = 0;
            currentPwmR = 0;
            pauseStartMs = millis();
            navState = NAV_PAUSE_AFTER_STRAIGHT;
            robotMode = MODE_RUN;
            Serial.println(">>> CMD,TURN: Chuan bi quay trai 90 do...");
        } else {
            Serial.println(">>> CMD,TURN: Dang quay hoac tam dung, bo qua.");
        }
        return;
    }

    // CMD,TRIM,<pwmL>,<pwmR> -> bù vận tốc 2 bánh chống lệch trực tiếp từ bàn phím Pi
    if (strcmp(name, "TRIM") == 0) {
        char* sL = strtok(NULL, ",");
        char* sR = strtok(NULL, ",");
        if (sL && sR) {
            trimPwmL = constrain(atoi(sL), -100, 100);
            trimPwmR = constrain(atoi(sR), -100, 100);
            // Đặt mốc lệch quãng đường hiện tại làm offset để PID encoder không chống lại trim
            noInterrupts();
            long cL = encLeft;
            long cR = encRight;
            interrupts();
            float dL = ((cL - startEncL) / leftPPR) * wheelCirc;
            float dR = ((cR - startEncR) / rightPPR) * wheelCirc;
            encDiffOffset = dL - dR;
            // Giữ nguyên targetHeading chuẩn để không làm méo quỹ đạo và phản hồi lái tức thì
            Serial.print(">>> TRIM cap nhat: L=");
            Serial.print(trimPwmL);
            Serial.print(", R=");
            Serial.print(trimPwmR);
            Serial.print(" | targetHeading=");
            Serial.println(targetHeading, 1);
        }
        return;
    }

    // CMD,DIST -> bỏ qua (không dùng quãng đường nữa)
    if (strcmp(name, "DIST") == 0) {
        return;
    }

    // CMD,KP,<val> -> đổi hệ số KP_HEADING
    if (strcmp(name, "KP") == 0) {
        char* sKp = strtok(NULL, ",");
        if (sKp) {
            KP_HEADING = atof(sKp);
            Serial.print(">>> Cap nhat KP_HEADING = ");
            Serial.println(KP_HEADING, 2);
        }
        return;
    }

    // CMD,KD,<val> -> đổi hệ số KD_HEADING
    if (strcmp(name, "KD") == 0) {
        char* sKd = strtok(NULL, ",");
        if (sKd) {
            KD_HEADING = atof(sKd);
            Serial.print(">>> Cap nhat KD_HEADING = ");
            Serial.println(KD_HEADING, 3);
        }
        return;
    }

    if (strcmp(name, "CLEAR") == 0) {
        overridePwmL = -999;
        overridePwmR = -999;
        noInterrupts();
        startEncL = encLeft;
        startEncR = encRight;
        interrupts();
        return;
    }

    // CMD,DRIVE,<pwmL>,<pwmR>
    if (strcmp(name, "DRIVE") == 0) {
        char* sL = strtok(NULL, ",");
        char* sR = strtok(NULL, ",");
        if (!sL || !sR) return;
        overridePwmL = constrain(atoi(sL), -255, 255);
        overridePwmR = constrain(atoi(sR), -255, 255);
        return;
    }

    // CMD,IR,LEFT / RIGHT / CLEAR
    if (strcmp(name, "IR") == 0) {
        char* val_str = strtok(NULL, ",");
        if (!val_str) return;
        const int IR_BOOST_PWM = 165;
        const int IR_WEAK_PWM  = 10;
        if (strcmp(val_str, "LEFT") == 0) {
            overridePwmL = IR_WEAK_PWM;
            overridePwmR = IR_BOOST_PWM;
        } else if (strcmp(val_str, "RIGHT") == 0) {
            overridePwmL = IR_BOOST_PWM;
            overridePwmR = IR_WEAK_PWM;
        } else if (strcmp(val_str, "CLEAR") == 0) {
            overridePwmL = -999;
            overridePwmR = -999;
        }
        return;
    }
}

void checkPiCommand() {
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            if (cmdBufIdx > 0) {
                cmdBuf[cmdBufIdx] = '\0';
                processPiCommand(cmdBuf);
                cmdBufIdx = 0;
            }
        } else if (cmdBufIdx < CMD_BUF_SIZE - 1) {
            cmdBuf[cmdBufIdx++] = c;
        }
    }
}

// ============================================================
//  Setup
// ============================================================
void setup() {
    Serial.begin(115200);    
    // Chờ mở Serial
    while (!Serial) {
        delay(10);
    }
    delay(500);
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    Serial.println("\n=== STM32 Robot – Thang / Quay trai (b) / Dung (Space) ===");

    pinMode(ENC_L_A, INPUT_PULLUP); pinMode(ENC_L_B, INPUT_PULLUP);
    pinMode(ENC_R_A, INPUT_PULLUP); pinMode(ENC_R_B, INPUT_PULLUP);
    lastLeftA  = digitalRead(ENC_L_A);
    lastRightA = digitalRead(ENC_R_A);
    attachInterrupt(digitalPinToInterrupt(ENC_L_A), isr_leftA,  CHANGE);
    attachInterrupt(digitalPinToInterrupt(ENC_R_A), isr_rightA, CHANGE);

    setupPWM();
    Wire.begin();
    delay(300);

    if (!bnoInit()) {
        Serial.println("!!! BNO055 FAILED!");
        while (true) { digitalWrite(LED_PIN, !digitalRead(LED_PIN)); delay(200); }
    }

    delay(5000);
    headingZeroRaw = readHeadingRaw();
    Serial.print(">>> Zero raw = "); Serial.print(headingZeroRaw, 1); Serial.println(" deg");

    digitalWrite(LED_PIN, HIGH); delay(300); digitalWrite(LED_PIN, LOW);
    noInterrupts();
    startEncL = encLeft;
    startEncR = encRight;
    interrupts();
    kfHeading.init(0.0f);
    targetHeading = 0.0f;
    navState = NAV_DRIVE_STRAIGHT;
    robotMode = MODE_STOP; // Chờ lệnh CMD,START từ Python
    Serial.println(">>> STM32 san sang! Cho lenh CMD,START tu Python.");
}

// ============================================================
//  Loop
// ============================================================
void loop() {
    checkPiCommand();

    static unsigned long lastIMU = 0;
    unsigned long now = millis();

    // Gửi $IMU định kỳ (heading chỉ để Pi đọc/log)
    if (now - lastIMU >= IMU_INTERVAL_MS) {
        lastIMU = now;
        sendIMUPacket();
    }

    if (robotMode == MODE_STOP) {
        setWheelPWM(0, 0);
        noInterrupts();
        prevEncL = encLeft;
        prevEncR = encRight;
        interrupts();
        lastControlMs = millis();
        return;
    }

    // Nếu Pi đã gửi CMD,DRIVE hoặc CMD,IR → dùng override đó
    if (overridePwmL != -999) {
        setWheelPWM(overridePwmL, overridePwmR);
        noInterrupts();
        startEncL = encLeft;
        startEncR = encRight;
        prevEncL  = encLeft;
        prevEncR  = encRight;
        interrupts();
        lastControlMs = millis();
        return;
    }

    // Chu kỳ điều khiển 20ms
    unsigned long nowMs = millis();
    float dt = (nowMs - lastControlMs) / 1000.0f;
    if (dt > 0.1f) dt = STRAIGHT_INTERVAL_MS / 1000.0f;

    if (dt >= (STRAIGHT_INTERVAL_MS / 1000.0f)) {
        lastControlMs = nowMs;

        long cL, cR;
        noInterrupts();
        cL = encLeft;
        cR = encRight;
        interrupts();

        // ============================================================
        // TẦNG 3: LỌC NHIỄU XUNG ENCODER & GIỚI HẠN VẬN TỐC VẬT LÝ
        // ============================================================
        float dL_raw = ((cL - prevEncL) / leftPPR) * wheelCirc;
        float dR_raw = ((cR - prevEncR) / rightPPR) * wheelCirc;
        prevEncL = cL;
        prevEncR = cR;

        // Vận tốc thô 2 bánh xe (mm/s)
        float vL_raw = dL_raw / dt;
        float vR_raw = dR_raw / dt;

        // 3a. Outlier Rejection: Loại bỏ các xung gai bất thường do nhiễu EMI
        if (fabs(vL_raw) > V_MAX_WHEEL_MM_S) {
            vL_raw = vL_filtered;
        }
        if (fabs(vR_raw) > V_MAX_WHEEL_MM_S) {
            vR_raw = vR_filtered;
        }

        // 3b. Low-Pass Filter (EMA) làm mịn vận tốc 2 bánh xe
        vL_filtered = 0.70f * vL_raw + 0.30f * vL_filtered;
        vR_filtered = 0.70f * vR_raw + 0.30f * vR_filtered;

        float dL = vL_filtered * dt;
        float dR = vR_filtered * dt;

        // Vận tốc góc thân xe từ encoder đã được lọc sạch (deg/s)
        float rateEnc = ((dL - dR) / WHEEL_BASE_MM) * (180.0f / 3.14159265f) / dt;

        // ============================================================
        // TẦNG 1 & TẦNG 2: ĐO GÓC THÔ & DUNG HỢP CẢM BIẾN (KALMAN)
        // ============================================================
        // Góc hướng đo từ BNO055 (đã qua fail-safe và lọc thông thấp ở Tầng 1)
        float bnoAngle = readHeadingMap();

        // Hợp nhất dữ liệu cảm biến bằng bộ lọc Kalman (Tầng 2)
        float filteredHeading = kfHeading.update(bnoAngle, rateEnc, dt);

        // ============================================================
        // TẦNG 4: BỘ ĐIỀU KHIỂN KHÁNG NHIỄU & LÀM ÊM CƠ CẤU CHẤP HÀNH
        // ============================================================
        switch (navState) {
            case NAV_DRIVE_STRAIGHT: {
                // Quãng đường tích luỹ của từng bánh từ lúc bắt đầu đoạn thẳng (mm)
                float totalDistL = ((cL - startEncL) / leftPPR) * wheelCirc;
                float totalDistR = ((cR - startEncR) / rightPPR) * wheelCirc;
                float distDiff   = (totalDistL - totalDistR) - encDiffOffset;

                // Tính toán PWM tăng tốc từ từ (Soft-Start tránh giật và trượt bánh)
                unsigned long driveElapsedMs = nowMs - straightStartMs;
                float currentBasePwm = (float)BASE_PWM;
                if (driveElapsedMs < (unsigned long)ACCEL_RAMP_MS) {
                    float progress = (float)driveElapsedMs / ACCEL_RAMP_MS;
                    currentBasePwm = START_PWM_MIN + progress * ((float)BASE_PWM - START_PWM_MIN);
                }

                // 4a. Sai số góc hướng so với targetHeading hiện tại
                float headingErr = wrap180(targetHeading - filteredHeading);

                // 4b. Deadzone: Vùng chết sai số góc để tránh hiện tượng đánh võng/lắc đuôi khi đã đi thẳng
                float effHeadingErr = headingErr;
                if (fabs(headingErr) <= HEADING_DEADZONE_DEG) {
                    effHeadingErr = 0.0f;
                }

                // 4c. Điều khiển PD: Thêm vi phân D trên yawRate để dập tắt dao động lắc lư
                float pTerm = KP_HEADING * effHeadingErr * GYRO_Z_SIGN;
                float dTerm = - (KD_HEADING * kfHeading.yawRate * GYRO_Z_SIGN);
                float steering = pTerm + dTerm - (K_ENC_DIFF * distDiff);

                // Giới hạn steering theo PWM hiện tại để tránh 1 bánh bị hãm quá mức lúc khởi động mềm
                float maxSteer = currentBasePwm * 0.45f;
                if (maxSteer < 6.0f) maxSteer = 6.0f;
                steering = constrain(steering, -maxSteer, maxSteer);

                int targetPwmL = constrain((int)(currentBasePwm + steering + trimPwmL), 0, 255);
                int targetPwmR = constrain((int)(currentBasePwm - steering + trimPwmR), 0, 255);

                // 4d. Slew Rate Limiter: Giới hạn gia tốc thay đổi PWM mỗi 20ms chống trượt bánh và giật motor
                currentPwmL = constrain(targetPwmL, currentPwmL - MAX_PWM_SLEW_STEP, currentPwmL + MAX_PWM_SLEW_STEP);
                currentPwmR = constrain(targetPwmR, currentPwmR - MAX_PWM_SLEW_STEP, currentPwmR + MAX_PWM_SLEW_STEP);

                setWheelPWM(currentPwmL, currentPwmR);
                break;
            }

            case NAV_PAUSE_AFTER_STRAIGHT: {
                setWheelPWM(0, 0);
                currentPwmL = 0;
                currentPwmR = 0;
                if (nowMs - pauseStartMs >= PAUSE_BETWEEN_STATES_MS) {
                    targetHeading = wrap180(targetHeading - TARGET_TURN_DEG);
                    navState = NAV_TURN_LEFT;
                    Serial.print(">>> [NAV] Quay trai 90 do. Huong muc tieu: ");
                    Serial.print(targetHeading, 1);
                    Serial.println(" deg");
                }
                break;
            }

            case NAV_TURN_LEFT: {
                float turnErr = wrap180(targetHeading - filteredHeading);

                if (fabs(turnErr) <= TURN_TOLERANCE_DEG) {
                    setWheelPWM(0, 0);
                    currentPwmL = 0;
                    currentPwmR = 0;
                    navState = NAV_PAUSE_AFTER_TURN;
                    pauseStartMs = nowMs;
                    Serial.print(">>> [NAV] Quay xong! Goc hien tai: ");
                    Serial.print(filteredHeading, 1);
                    Serial.println(" deg");
                } else {
                    // PD controller quay góc tại chỗ với giảm tốc mượt
                    float absErr = fabs(turnErr);
                    float pTerm = absErr * KP_TURN;
                    float dTerm = - (0.04f * fabs(kfHeading.yawRate));
                    int turnPwm = constrain((int)(pTerm + dTerm), TURN_PWM_MIN, TURN_PWM_MAX);

                    if (turnErr < 0) {
                        setWheelPWM(-turnPwm, turnPwm);
                    } else {
                        setWheelPWM(turnPwm, -turnPwm);
                    }
                }
                break;
            }

            case NAV_PAUSE_AFTER_TURN: {
                setWheelPWM(0, 0);
                currentPwmL = 0;
                currentPwmR = 0;
                if (nowMs - pauseStartMs >= PAUSE_BETWEEN_STATES_MS) {
                    noInterrupts();
                    startEncL = encLeft;
                    startEncR = encRight;
                    prevEncL  = encLeft;
                    prevEncR  = encRight;
                    interrupts();
                    encDiffOffset = 0.0f;
                    trimPwmL = 0;
                    trimPwmR = 0;
                    vL_filtered = 0.0f;
                    vR_filtered = 0.0f;
                    straightStartMs = millis();
                    lastControlMs   = millis();
                    navState = NAV_DRIVE_STRAIGHT;
                    Serial.print(">>> [NAV] Di tiep thang, huong ");
                    Serial.print(targetHeading, 1);
                    Serial.println(" deg");
                }
                break;
            }
        }
    }
}
