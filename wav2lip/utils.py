import cv2
import numpy as np
from pypinyin import Style
from pypinyin.converter import UltimateConverter
from pypinyin.core import Pinyin
from skimage import transform as trans


arcface_src = np.array([[38.2946, 51.6963],
                        [73.5318, 51.5014],
                        [56.0252, 71.7366],
                        [41.5493, 92.3655],
                        [70.7299, 92.2041]], dtype=np.float32)
arcface_src = np.expand_dims(arcface_src, axis=0)


def estimate_norm(lmk, face_size, dst_face_size, expand_size):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1) 
    min_M = []                                              
    min_index = []                                          
    min_error = float('inf')   

    assert face_size == 112
    src = (arcface_src / face_size * dst_face_size) + (expand_size - dst_face_size) / 2                
   
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))

        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def metrix_M(face_size, expand_size, keypoints=None):
    id_size = 112
    detected_lmk = np.concatenate(keypoints).reshape(5, 2)
    M, _ = estimate_norm(detected_lmk, id_size, face_size, expand_size)
    Minv = np.identity(3, dtype=np.single)
    Minv[0:2, :] = M
    M = Minv[0:2, :]
    return M   


def decompose_tfm(tfm):
    tfm = tfm.copy()
    s_x = np.sqrt(tfm[0][0] ** 2 + tfm[0][1] ** 2)
    s_y = np.sqrt(tfm[1][0] ** 2 + tfm[1][1] ** 2)

    t_x = tfm[0][2]
    t_y = tfm[1][2]

    #平移旋转矩阵rt
    rt = np.array([
        [tfm[0][0] / s_x, tfm[0][1] / s_x, t_x / s_x],
        [tfm[1][0] / s_y, tfm[1][1] / s_y, t_y / s_y],
    ])

    #缩放矩阵s
    s = np.array([
        [s_x, 0, 0],
        [0, s_y, 0]
    ])

    # _rt = np.vstack([rt, [[0, 0, 1]]])
    # _s = np.vstack([s, [[0, 0, 1]]])
    # print(np.dot(_s, _rt)[:2] - tfm)

    return rt, s


def img_warp(img, M, expand_size, adjust=0):
    warped = cv2.warpAffine(img, M, (expand_size, expand_size))
    warped = warped - np.uint8(adjust)
    warped = np.clip(warped, 0, 255)
    return warped


def img_warp_back_inv_m(img, img_to, inv_m):
    h_up, w_up, c = img_to.shape

    mask = np.ones_like(img).astype(np.float32)
    inv_mask = cv2.warpAffine(mask, inv_m, (w_up, h_up))
    inv_img = cv2.warpAffine(img, inv_m, (w_up, h_up))

    img_to[inv_mask == 1] = inv_img[inv_mask == 1]
    return img_to


def get_video_fps(vfile):
    cap = cv2.VideoCapture(vfile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


class laplacianSmooth(object):

    def __init__(self, smoothAlpha=0.3):
        self.smoothAlpha = smoothAlpha
        self.pts_last = None

    def smooth(self, pts_cur):
        if self.pts_last is None:
            self.pts_last = pts_cur.copy()
            return pts_cur.copy()
        x1 = min(pts_cur[:, 0])
        x2 = max(pts_cur[:, 0])
        y1 = min(pts_cur[:, 1])
        y2 = max(pts_cur[:, 1])
        width = x2 - x1
        pts_update = []
        for i in range(len(pts_cur)):
            x_new, y_new = pts_cur[i]
            x_old, y_old = self.pts_last[i]
            tmp = (x_new - x_old) ** 2 + (y_new - y_old) ** 2
            w = np.exp(-tmp / (width * self.smoothAlpha))
            x = x_old * w + x_new * (1 - w)
            y = y_old * w + y_new * (1 - w)
            pts_update.append([x, y])
        pts_update = np.array(pts_update)
        self.pts_last = pts_update.copy()

        return pts_update

def lazy_pinyin(hans, style=Style.NORMAL, errors='default', strict=True,
                v_to_u=False, neutral_tone_with_five=False, tone_sandhi=False):
    """将汉字转换为拼音，返回不包含多音字结果的拼音列表.

    与 :py:func:`~pypinyin.pinyin` 的区别是返回的拼音是个字符串，
    并且每个字只包含一个读音.

    :param hans: 汉字字符串( ``'你好吗'`` )或列表( ``['你好', '吗']`` ).
                 可以使用自己喜爱的分词模块对字符串进行分词处理,
                 只需将经过分词处理的字符串列表传进来就可以了。
    :type hans: unicode 字符串或字符串列表
    :param style: 指定拼音风格，默认是 :py:attr:`~pypinyin.Style.NORMAL` 风格。
                  更多拼音风格详见 :class:`~pypinyin.Style`。
    :param errors: 指定如何处理没有拼音的字符，详情请参考
                   :py:func:`~pypinyin.pinyin`
    :param strict: 只获取声母或只获取韵母相关拼音风格的返回结果
                   是否严格遵照《汉语拼音方案》来处理声母和韵母，
                   详见 :ref:`strict`
    :param v_to_u: 无声调相关拼音风格下的结果是否使用 ``ü`` 代替原来的 ``v``
                   当为 False 时结果中将使用 ``v`` 表示 ``ü``
    :type v_to_u: bool
    :param neutral_tone_with_five: 声调使用数字表示的相关拼音风格下的结果是否
                                   使用 5 标识轻声
    :type neutral_tone_with_five: bool
    :param tone_sandhi: 是否按照声调 `变调规则 <https://en.wikipedia.org/wiki/Standard_Chinese_phonology#Tone_sandhi>`__
                        对拼音进行处理
                        （使用预先通过分词库进行过分词后的结果作为 ``hans``
                        参数的值效果会更好，因为变调效果依赖分词效果）
    :type tone_sandhi: bool
    :return: 拼音列表(e.g. ``['zhong', 'guo', 'ren']``)
    :rtype: list

    :raise AssertionError: 当传入的字符串不是 unicode 字符时会抛出这个异常

    Usage::

      >>> from pypinyin import lazy_pinyin, Style
      >>> import pypinyin
      >>> lazy_pinyin('中心')
      ['zhong', 'xin']
      >>> lazy_pinyin('中心', style=Style.TONE)
      ['zhōng', 'xīn']
      >>> lazy_pinyin('中心', style=Style.FIRST_LETTER)
      ['z', 'x']
      >>> lazy_pinyin('中心', style=Style.TONE2)
      ['zho1ng', 'xi1n']
      >>> lazy_pinyin('中心', style=Style.CYRILLIC)
      ['чжун1', 'синь1']
      >>> lazy_pinyin('战略', v_to_u=True)
      ['zhan', 'lüe']
      >>> lazy_pinyin('衣裳', style=Style.TONE3, neutral_tone_with_five=True)
      ['yi1', 'shang5']
      >>> lazy_pinyin('你好', style=Style.TONE2, tone_sandhi=True)
      ['ni2', 'ha3o']
    """  # noqa
    _pinyin = Pinyin(UltimateConverter(
        v_to_u=v_to_u, neutral_tone_with_five=neutral_tone_with_five,
        tone_sandhi=tone_sandhi))
    return _pinyin.lazy_pinyin(
        hans, style=style, errors=errors, strict=strict)