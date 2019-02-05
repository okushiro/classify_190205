//
//  ViewController.swift
//  classify
//
//  Created by 奥城健太郎 on 2019/02/05.
//  Copyright © 2019 奥城健太郎. All rights reserved.
//

import UIKit
import CoreML
import Vision

//画像分類（写真）
class ViewController: UIViewController,
    UINavigationControllerDelegate,
    UIImagePickerControllerDelegate,
UIGestureRecognizerDelegate {
    //UI
    
    @IBOutlet weak var imageView: UIImageView!
    
    
    @IBOutlet weak var lblText: UILabel!
    
    //(2)モデルの生成
    var model = try! VNCoreMLModel(for: MobileNet().model)
    
    
    //====================
    //ライフサイクル
    //====================
    //ビュー表示時に呼ばれる
    override func viewDidAppear(_ animated: Bool) {
        //アクションシートの表示
        if self.imageView.image == nil {
            showActionSheet()
        }
    }
    
    
    //====================
    //イベント
    //====================
    //画面タッチ時に呼ばれる
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        showActionSheet()
    }
    
    
    //====================
    //アクションシート
    //====================
    //アクションシートの表示
    func showActionSheet() {
        let actionSheet = UIAlertController(title: nil, message: nil,
                                            preferredStyle: .actionSheet)
        actionSheet.addAction(UIAlertAction(title: "カメラ", style: .default) {
            action in
            self.openPicker(sourceType: .camera)
        })
        actionSheet.addAction(UIAlertAction(title: "フォトライブラリ", style: .default) {
            action in
            self.openPicker(sourceType: .photoLibrary)
        })
        actionSheet.addAction(UIAlertAction(title: "キャンセル", style: .cancel))
        self.present(actionSheet, animated: true, completion: nil)
    }
    
    
    //====================
    //アラート
    //====================
    //アラートの表示
    func showAlert(_ text: String!) {
        let alert = UIAlertController(title: text, message: nil,
                                      preferredStyle: UIAlertController.Style.alert)
        alert.addAction(UIAlertAction(title: "OK",
                                      style: UIAlertAction.Style.default, handler: nil))
        self.present(alert, animated: true, completion: nil)
    }
    
    
    //====================
    //(1)イメージピッカー
    //====================
    //イメージピッカーのオープン
    func openPicker(sourceType: UIImagePickerController.SourceType) {
        let picker = UIImagePickerController()
        picker.sourceType = sourceType
        picker.delegate = self
        self.present(picker, animated: true, completion: nil)
    }
    
    //イメージピッカーのイメージ取得時に呼ばれる
    func imagePickerController(_ picker: UIImagePickerController,
                               didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        //イメージの取得
        var image = info[UIImagePickerController.InfoKey.originalImage] as! UIImage
        
        //画像向きの補正
        let size = image.size
        UIGraphicsBeginImageContext(size)
        image.draw(in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        image = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        
        //イメージの指定
        self.imageView.image = image
        
        //クローズ
        picker.presentingViewController!.dismiss(animated:true, completion:nil);
        
        //予測
        predict(image);
    }
    
    //イメージピッカーのキャンセル時に呼ばれる
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        //クローズ
        picker.presentingViewController!.dismiss(animated:true, completion:nil)
    }
    
    
    //====================
    //画像分類
    //====================
    //(3)予測
    func predict(_ image: UIImage) {
        DispatchQueue.global(qos: .default).async {
            //リクエストの生成
            let request = VNCoreMLRequest(model: self.model) {
                request, error in
                //エラー処理
                if error != nil {
                    self.showAlert(error!.localizedDescription)
                    return
                }
                
                //検出結果の取得
                let observations = request.results as! [VNClassificationObservation]
                var text: String = "\n"
                for i in 0..<min(3, observations.count) { //上位3件
                    let probabillity = Int(observations[i].confidence*100) //信頼度
                    let label = observations[i].identifier //ラベル
                    text += "\(label) : \(probabillity)%\n"
                }
                
                //UIの更新
                DispatchQueue.main.async {
                    self.lblText.text = text
                }
            }
            
            //入力画像のリサイズ指定
            request.imageCropAndScaleOption = .centerCrop
            
            //UIImageをCIImageに変換
            let ciImage = CIImage(image: image)!
            
            //画像の向きの取得
            let orientation = CGImagePropertyOrientation(
                rawValue: UInt32(image.imageOrientation.rawValue))!
            
            //ハンドラの生成と実行
            let handler = VNImageRequestHandler(
                ciImage: ciImage, orientation: orientation)
            guard (try? handler.perform([request])) != nil else {return}
        }
    }
}
