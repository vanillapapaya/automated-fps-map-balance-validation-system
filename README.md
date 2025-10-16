# 절차적 맵 검증 시스템

팀 간 시야 노출 비율의 수학적 검증을 통해 게임플레이 균형을 보장하는 제약 기반 절차적 맵 생성 시스템입니다.

## 개요

이 시스템은 Perlin noise를 사용하여 지형 heightmap을 생성하고, 공간 분석 메트릭을 통해 경쟁 팀 간의 전술적 공정성을 검증합니다. 팀 존 간의 시야 노출 비율에 관한 엄격한 균형 기준을 충족하는 맵만 허용됩니다.

## 주요 기능

- Perlin noise 기반 절차적 지형 생성
- 픽셀 단위 line-of-sight 분석을 통한 정확한 노출도 계산
- 규칙 기반 검증 엔진
- 단계별 생성 및 분석 과정 시각화

## 시야 노출 분석

시스템은 팀 존 간 line-of-sight 체크를 수행하여 시야 노출 비율을 계산합니다:

- **샘플링 모드**: 대표 샘플 포인트(존당 50-100개)를 사용한 빠른 근사 계산
- **전수 조사 모드**: 최대 정확도를 위한 픽셀 단위 전수 계산

각 타겟 존의 픽셀에 대해, 노출 비율은 해당 픽셀을 볼 수 있는 관찰자 픽셀의 백분율을 나타냅니다.

## 설치

### 요구사항

- Python 3.8+
- uv (권장) 또는 pip

### 설치 방법

```bash
# uv 사용 (권장)
uv venv
uv pip install -r requirements.txt

# pip 사용
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 사용법

### 빠른 시작

검증된 맵과 시각화 생성:

```bash
uv run python examples/visualize_steps.py
```

### 전수 조사 테스트

픽셀 단위 전수 조사 분석 테스트:

```bash
uv run python examples/test_full_analysis.py
```

### Jupyter Notebook

대화형 분석을 위한 Jupyter notebook:

```bash
jupyter notebook notebooks/step_by_step_analysis.ipynb
```

Notebook은 다음과 같은 단계별 분석을 제공합니다:

1. **파라미터 설정**: 맵 크기, Perlin noise 파라미터, 분석 모드 선택
2. **원본 Heightmap 생성**: Perlin noise를 사용한 자연스러운 지형 생성
3. **팀 존 할당**: 맵을 Team A와 Team B 구역으로 분할
4. **샘플 포인트 배치**: 가시성 분석을 위한 대표 샘플 포인트 생성
5. **Line-of-Sight 데모**: 지형 장애물을 고려한 가시성 분석 시연
6. **노출도 계산**: 픽셀 단위 전수 조사를 통한 정확한 노출도 계산 및 히트맵 생성
7. **밸런스 검증**: 사전 정의된 검증 규칙을 적용하여 맵 공정성 판단

Notebook은 각 단계를 실시간으로 시각화하며, 픽셀별 노출도 통계, 위험 지역 및 안전 지역 분석을 포함합니다. 전수 조사 모드에서는 각 픽셀의 정확한 노출 값을 계산하여 히트맵으로 표시합니다. 마지막 검증 단계는 텍스트 리포트로 생성됩니다.

## 설정

### 맵 파라미터

- `width`, `height`: 맵 크기 (픽셀 단위)
- `octaves`: Perlin noise 옥타브 수 (디테일 수준 조절)
- `persistence`: 옥타브 간 진폭 감쇠율
- `lacunarity`: 옥타브 간 주파수 스케일링
- `scale`: 전체 노이즈 스케일
- `seed`: 재현성을 위한 랜덤 시드

### 분석 파라미터

- `observer_height`: line-of-sight 계산을 위한 높이 오프셋 (0-1 정규화)
- `use_full_analysis`: 샘플링과 전수 조사 간 전환
- `num_samples`: 존당 샘플 포인트 수 (샘플링 모드 전용)

### 검증 규칙

기본 규칙:

1. 최대 노출 임계값 (< 0.4)
2. 팀 간 노출 균형 (차이 < 0.15)
3. 최소 평균 노출 (> 0.1)
4. 존 커버리지 요구사항 (> 0.8)
5. 지형 변화 기준 (표준편차 > 0.05)

## 성능 고려사항

### 맵 크기별 분석 시간

| 크기 | 샘플링 모드 | 전수 조사 | LOS 체크 수 |
|------|-------------|-----------|-------------|
| 64x64 | < 1초 | 2-5분 | ~2M |
| 128x128 | < 2초 | 1-2시간 | ~64M |
| 256x256 | < 5초 | 20시간+ | ~1B |

전수 조사는 최종 검증에만 권장됩니다. 반복 개발 중에는 샘플링 모드를 사용하십시오.

## 출력

시스템은 6개의 단계별 출력을 생성합니다:

1. 원본 heightmap (Perlin noise 지형) - PNG
2. 팀 존 할당 - PNG
3. 샘플 포인트 분포 - PNG
4. Line-of-sight 계산 - PNG
5. 노출도 분석 히트맵 - PNG
6. 검증 결과 리포트 - TXT

출력 파일은 `output/` 디렉토리에 저장됩니다.

## 프로젝트 구조

```
Procedural-map-validation-system/
├── src/
│   ├── map_generator.py       # 지형 생성
│   ├── spatial_analyzer.py    # 가시성 분석
│   ├── validation_engine.py   # 규칙 검증
│   ├── step_visualizer.py     # 시각화
│   └── logger.py              # 로깅 설정
├── examples/
│   ├── visualize_steps.py     # 전체 시각화 파이프라인
│   └── test_full_analysis.py  # 전수 조사 테스트
├── notebooks/
│   └── step_by_step_analysis.ipynb  # 대화형 분석
├── config/
│   └── validation_rules.json  # 검증 규칙
└── README.md
```

## 검증 메트릭

시스템은 다음과 같은 주요 메트릭을 계산합니다:

- `team_a_exposure`: Team B 관찰자에 대한 Team A 존의 평균 노출도
- `team_b_exposure`: Team A 관찰자에 대한 Team B 존의 평균 노출도
- `exposure_difference`: 팀 간 노출도의 절대 차이
- `max_exposure`: 두 팀 중 최대 노출 값
- `avg_exposure`: 두 팀의 평균 노출도

균형잡힌 맵의 일반적인 기준:
- 노출도 차이 < 0.15
- 개별 팀 노출도 < 0.4
- 평균 노출도 0.1 ~ 0.4 사이

## 알고리즘 세부사항

### Line-of-Sight 알고리즘

두 점 간의 가시성을 확인하기 위해 Bresenham의 직선 알고리즘 사용:

1. 관찰자와 타겟 간의 직선 상 점들 생성
2. 각 점에서 차단되지 않은 예상 높이 계산
3. 실제 지형 고도와 비교
4. 지형이 예상 높이를 초과하면 차단으로 반환

### 노출도 계산

전수 조사 모드의 픽셀 단위 분석:

```python
for each target_pixel in target_zone:
    visible_count = 0
    for each observer_pixel in observer_zone:
        if line_of_sight(observer_pixel, target_pixel):
            visible_count += 1
    exposure[target_pixel] = visible_count / total_observers
```

팀 노출도는 해당 팀 존 내 모든 픽셀 노출도의 평균입니다.

## 노출도 해석

픽셀별 노출도 값의 의미:

- **0.0**: 전혀 보이지 않음 (완전 엄폐, 안전)
- **0.0 ~ 0.3**: 낮은 노출 (방어적 지역, 안전)
- **0.3 ~ 0.5**: 중간 노출 (균형잡힌 지역)
- **0.5 ~ 1.0**: 높은 노출 (위험 지역, 공격적)
- **1.0**: 완전히 노출됨 (매우 위험)

전수 조사 모드에서는 각 픽셀의 정확한 노출 값을 계산하여:
- 위험 지역 (노출도 > 0.5) 픽셀 수
- 안전 지역 (노출도 < 0.3) 픽셀 수
- 노출도 범위 (최소 ~ 최대)

등의 상세 통계를 제공합니다.

## 라이선스

MIT License

## 참고자료

상세한 프로젝트 문서 및 설계 결정 사항은 `CLAUDE.md`를 참조하십시오.
