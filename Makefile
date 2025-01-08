TARGET := run
YOLOV8_PATH ?= /datas/develop/PycharmProject/yolov8-seg/runs/train/yolov8m-seg-2/weights/best.pt
YOLOV8_BIN_MODEL_PATH ?= yolov8.bin
PYTHON ?= /opt/miniconda3/bin/python
CC ?= gcc # or clang
CFLAGS = -g
LDFLAGS = 
LDLIBS += -L./usrLib -lm

SRCS += $(wildcard *.c)
OBJS := $(patsubst %.c, %.o, ${SRCS})
MODULES_SRCS += $(wildcard modules/*.c)
MODULES_OBJS := $(patsubst modules/%.c, modules/%.o, ${MODULES_SRCS})

$(info $(MODULES_OBJS))

########## Hf Model ###########
${YOLOV8_PATH}:
	@if [ ! -f "${YOLOV8_PATH}" ]; then \
		echo "Error: ${YOLOV8_PATH} does not exist!"; \
		exit 1; \
	fi

########## Export bin Model ###########
${YOLOV8_BIN_MODEL_PATH}: ${YOLOV8_PATH}
	${PYTHON} export/export_yolov8_bin.py ${YOLOV8_BIN_MODEL_PATH} --checkpoint=${YOLOV8_PATH};
	

######### Define some names for so #######
all: ${YOLOV8_BIN_MODEL_PATH} $(TARGET) 

########## Shared lib ###########
$(TARGET): ${MODULES_OBJS} $(OBJS) 
	${CC} ${CFLAGS} -fopenmp ${MODULES_OBJS} $(OBJS) -o $(TARGET) $(LDFLAGS) ${LDLIBS}

########## Shared lib ###########
%.o: %.c
	${CC} ${CFLAGS} -fopenmp -c $< -o $@ -lm

# modules/%.o: modules/%.c
# 	${CC} ${CFLAGS} -fopenmp -c $< -o $@ -lm

clean:
	rm -f *.o
	rm -f modules/*.o
	rm -f $(TARGET)
	rm -f yolov8.bin