# Seminar 04. Итоговый отчет

В папке собраны финальные файлы работы:

- `source_video_2026-06-01_03-38-58.mp4` - исходное видео объекта.
- `final_model_dense_object_poisson_clean_largest.glb` - итоговая 3D-модель.
- `Video_to_3D_COLMAP_executed.ipynb` - выполненный ноутбук реконструкции по видео.
- `Image_to_Voxel_executed.ipynb` - выполненный ноутбук обучения Image-to-Voxel моделей.

## Video-to-3D reconstruction

Пайплайн: видео -> отбор резких кадров -> маски объекта -> COLMAP sparse SfM -> COLMAP dense MVS -> фильтрация dense-облака по маскам -> построение сетки -> экспорт `GLB`.

Основные результаты:

| Метрика | Значение |
|---|---:|
| Проверено кадров-кандидатов | 420 |
| Использовано кадров | 240 |
| Зарегистрировано изображений COLMAP | 240 |
| Sparse-точек COLMAP | 23 372 |
| Sparse-точек объекта | 8 645 |
| Dense-точек COLMAP | 2 473 408 |
| Dense-точек объекта | 1 246 059 |
| Вершин в итоговой Poisson-модели | 876 735 |
| Граней в итоговой Poisson-модели | 1 749 097 |

Итоговый вид модели:

![Итоговая 3D-модель](assets/final_model_render.png)

Кадры и маски, использованные для реконструкции:

![Кадры и маски](assets/video_selected_frames_and_masks.png)

Dense-облако объекта:

![Dense object points](assets/video_dense_object_points_preview.png)

Превью итоговой Poisson-сетки:

![Poisson mesh preview](assets/video_poisson_clean_preview.png)

Альтернативная BPA-сетка была дополнительно построена как более открытая модель поверхности:

![BPA mesh preview](assets/video_bpa_clean_preview.png)

## Image-to-Voxel

В ноутбуке подготовлены пары `PNG + STL`, реализованы attention-блоки и обучены три модели для восстановления voxel-grid по одному изображению.

| Модель | Val Dice | Val IoU | Test Dice | Test IoU | Voxel accuracy |
|---|---:|---:|---:|---:|---:|
| `model_1_cnn` | 0.7801 | 0.6607 | 0.7984 | 0.6812 | 0.9098 |
| `model_2_se_attention` | 0.7889 | 0.6742 | 0.8071 | 0.6938 | 0.9112 |
| `model_3_self_attention` | 0.7780 | 0.6594 | 0.7964 | 0.6818 | 0.9052 |

Лучшая модель по test Dice: `model_2_se_attention`, `Test Dice = 0.8071`, `Test IoU = 0.6938`.

Графики обучения:

![Image-to-Voxel training curves](assets/image_to_voxel_training_curves.png)

Примеры предсказаний:

![Image-to-Voxel sample predictions](assets/image_to_voxel_sample_predictions.png)

