import pygame
import sys
import math

# 初始化Pygame
pygame.init()

# 窗口设置
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("半圆盘指针控制器")

# 颜色定义
BACKGROUND = (40, 44, 52)
DISK_COLOR = (86, 156, 214)
POINTER_COLOR = (220, 163, 100)
TEXT_COLOR = (220, 220, 220)
HIGHLIGHT = (152, 195, 121)

# 半圆盘参数
center_x, center_y = WIDTH // 2, HEIGHT // 2 + 100
radius = 200
pointer_length = 180

# 初始角度（0度指向左侧）
current_angle = 0

# 字体设置
font = pygame.font.SysFont(None, 36)
key_font = pygame.font.SysFont(None, 32)

# 主循环
clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            # 按键控制
            if event.key == pygame.K_a:
                current_angle = 180    # 左0度
            elif event.key == pygame.K_s:
                current_angle = 135   # 左前45度
            elif event.key == pygame.K_d:
                current_angle = 90  # 右前135度
            elif event.key == pygame.K_f:
                current_angle = 45  # 右180度
            elif event.key == pygame.K_g:
                current_angle = 0    # 上90度

    # 清屏
    screen.fill(BACKGROUND)
    
    # 绘制半圆盘
    pygame.draw.arc(screen, DISK_COLOR, 
                   (center_x - radius, center_y - radius, radius * 2, radius * 2),
                   math.pi, 2 * math.pi, 8)
    
    # 绘制角度标记
    for angle in [0, 45, 90, 135, 180]:
        rad_angle = math.radians(angle)
        start_x = center_x + (radius - 20) * math.cos(rad_angle)
        start_y = center_y - (radius - 20) * math.sin(rad_angle)
        end_x = center_x + radius * math.cos(rad_angle)
        end_y = center_y - radius * math.sin(rad_angle)
        pygame.draw.line(screen, DISK_COLOR, (start_x, start_y), (end_x, end_y), 3)
        
        # 绘制角度标签
        label_x = center_x + (radius + 20) * math.cos(rad_angle)
        label_y = center_y - (radius + 20) * math.sin(rad_angle)
        text = font.render(str(angle) + "°", True, TEXT_COLOR)
        text_rect = text.get_rect(center=(label_x, label_y))
        screen.blit(text, text_rect)
    
    # 绘制中心点
    pygame.draw.circle(screen, POINTER_COLOR, (center_x, center_y), 10)
    
    # 计算指针终点坐标
    rad_angle = math.radians(current_angle)
    end_x = center_x + pointer_length * math.cos(rad_angle)
    end_y = center_y - pointer_length * math.sin(rad_angle)
    
    # 绘制指针
    pygame.draw.line(screen, POINTER_COLOR, (center_x, center_y), 
                    (end_x, end_y), 8)
    pygame.draw.circle(screen, POINTER_COLOR, (int(end_x), int(end_y)), 12)
    
    # 绘制当前角度信息
    angle_text = font.render(f"Angle: {current_angle}°", True, HIGHLIGHT)
    screen.blit(angle_text, (WIDTH // 2 - angle_text.get_width() // 2, 50))
    
    # 绘制控制说明
    # controls = [
    #     "键盘控制:",
    #     "A 键 - 左0度",
    #     "S 键 - 左前45度",
    #     "D 键 - 右前135度",
    #     "F 键 - 右180度"
    # ]
    
    # for i, text in enumerate(controls):
    #     color = HIGHLIGHT if i > 0 and text[0] in ['A','S','D','F'] else TEXT_COLOR
    #     ctrl_text = key_font.render(text, True, color)
    #     screen.blit(ctrl_text, (WIDTH // 2 - ctrl_text.get_width() // 2, 100 + i * 40))
    
    # 更新屏幕
    pygame.display.flip()
    clock.tick(60)

# 退出Pygame
pygame.quit()
sys.exit()