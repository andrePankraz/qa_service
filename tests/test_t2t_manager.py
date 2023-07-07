"""
This file was created by ]init[ AG 2022.

Tests for Whatever.
"""
import logging
from qa_service.t2t_manager.manager import T2tManager

log = logging.getLogger(__name__)

t2tManager = T2tManager()


def test_question_elterngeld():
    log.debug(
        t2tManager.generate(
            """Frage: Wie wirkt sich der Brexit auf den Anspruch auf Elterngeld für britische Staatsangehörige aus?

Nutze für die Beantwortung ausschließlich folgende relevanteste Fakten (ansonsten antworte mit [NIX]):
Kann ich Elterngeld erhalten, obwohl ich in unterschiedlichen Ländern lebe und arbeite? Wenn beide Eltern in verschiedenen Ländern arbeiten, dann bekommen beide Eltern Familien-Leistungen vorrangig von dem Land, in dem auch das Kind wohnt. Seit dem sogenannten Brexit gehört Großbritannien nicht mehr zur EU. Falls Sie die britische Staatsangehörigkeit haben und Grenzgängerin oder Grenzgänger sind, können Sie unter bestimmten Voraussetzungen trotzdem Elterngeld nach den besonderen EU-Regelungen bekommen. Für weitere Informationen wenden Sie sich bitte an Ihre Elterngeldstelle.

Antwort:"""
        )
    )


def test_question_neustarthilfe():
    log.debug(
        t2tManager.generate(
            """### Instruction:
Du agierst als eine Suchmaschine, spezialisiert auf das Antragsportal "Überbrückungshilfen" mit Fokus auf das Teilprogramm "Neustarthilfe". Du beantwortest Fragen zu diesem Thema basierend auf gegebenem Kontext.

Regeln:

1) Nutze nur den im Kontext gegebenen Fakten für die Antwort, füge kein anderes Wissen hinzu.
2) ZITIERE die Referenzangabe [[x]] für jeden verwendeten Fakt.
3) Ignoriere irrelevante Fakten und Fragen im Kontext.
4) Antwortet nur auf Fragen im Themenbereich "Neustarthilfe".
5) Wenn kein passender Fakt vorhanden ist, antworte mit: 'Ich bin mir nicht sicher.' und verweise auf ähnliche Fragen, wenn möglich.
6) Deine Antwort sollte MAXIMAL 250 Wörter haben und in verständlicher und einfacher Sprache verfasst sein.
7) Verifiziere deine Antwort sorgfältig.

Kontext:
[[2]] Wer ist antragsberechtigt? Ein bereits gestellter oder noch zu stellender Antrag auf die Neustarthilfe 2022 für das erste Quartal ist keine Voraussetzung für die Beantragung der Neustarthilfe 2022 für das zweite Quartal. Daher ist es auch möglich, die Neustarthilfe 2022 nur für das zweite Quartal zu beantragen, siehe Ziffer 4.1. Für die Neustarthilfe 2022 grundsätzlich antragsberechtigt sind selbständig erwerbstätige Soloselbständige, Kapitalgesellschaften und Genossenschaften (im Folgenden zusammen mit den Soloselbständigen: Antragstellende) aller Branchen, wenn sie
- als Soloselbständige ihre selbständige Tätigkeit im Haupterwerb ausüben, das heißt dass der überwiegende Teil der Summe ihrer Einkünfte (mindestens 51 Prozent) aus einer gewerblichen (§ 15 Einkommenssteuergesetz, EStG) und/oder freiberuflichen (§ 18 EStG) Tätigkeit stammt (vergleiche auch 2.4), oder
    als Ein-Personen-Kapitalgesellschaft den überwiegenden Teil der Summe der Einkünfte (mindestens 51 Prozent) aus vergleichbaren Tätigkeiten (vergleiche 2.2, 2.4) erzielen und die Gesellschafterin oder der Gesellschafter 100 Prozent der Geschäftsanteile an der Ein-Personen-Kapitalgesellschaft hält und mindestens 20 Stunden pro Woche von dieser beschäftigt wird oder
    als Mehr-Personen-Kapitalgesellschaft den überwiegenden Teil ihrer Einkünfte (mindestens 51 Prozent) aus vergleichbaren Tätigkeiten (vergleiche 2.2, 2.4) erzielen und mindestens eine oder einer der Gesellschafterinnen oder Gesellschafter 25 Prozent oder mehr der Gesellschaftsanteile hält und mindestens 20 Stunden pro Woche von der Gesellschaft beschäftigt wird oder
    als Genossenschaft den überwiegenden Teil ihrer Einkünfte (mindestens 51 Prozent) aus vergleichbaren Tätigkeiten erzielen und mindestens ein Mitglied mindestens 20 Stunden pro Woche von der Genossenschaft beschäftigt wird und die Genossenschaft insgesamt nicht mehr als zehn Angestellte (Vollzeit-Äquivalent, Mitglieder und Nicht-Mitglieder) beschäftigt, wobei Angestellte, die nicht Mitglieder sind, weniger als ein Vollzeit-Äquivalent ausmachen müssen (siehe nächste Ziffer sowie Ziffer 2.3),
- weniger als eine Angestellte oder einen Angestellten (Vollzeit-Äquivalent) beschäftigen, die oder der nicht Gesellschafterin oder Gesellschafter oder Mitglied der oder des Antragstellenden ist (vergleiche Ziffer 2.5),
- bei einem deutschen Finanzamt für steuerliche Zwecke erfasst sind,
- ihre selbständige Geschäftstätigkeit vor dem 1. Oktober 2021 aufgenommen haben beziehungsweise vor dem 1. Oktober 2021 gegründet wurden und
- keine Fixkostenerstattung in der Überbrückungshilfe IV beantragt oder erhalten haben und noch keine Neustarthilfe 2022 beantragt oder erhalten haben; siehe im Einzelnen hierzu die untenstehenden Hinweise. Nicht antragsberechtigt sind Antragstellende (Ausschlusskriterien), die
- sich bereits zum 31. Dezember 2019 in wirtschaftlichen Schwierigkeiten befunden haben und diesen Status danach nicht wieder überwunden haben,
- ihre Geschäftstätigkeit dauerhaft eingestellt oder ein nationales Insolvenzverfahren beantragt oder eröffnet haben. Begünstigte oder teil-begünstigte Direktantragstellende der Neustarthilfe (Förderzeitraum Januar bis Juni 2021) sind für die Neustarthilfe 2022 für das zweite Quartal (Förderzeitraum 1. April bis 30. Juni 2022) nur dann antragsberechtigt, wenn den zuständigen Bewilligungsstellen die Selbsterklärung zur Endabrechnung der Neustarthilfe (siehe FAQs Neustarthilfe Ziffer 4.8) vorliegt. Kurz befristete Beschäftigungsverhältnisse in den Darstellenden Künsten (bis zu 14 Wochen) sowie unständige Beschäftigungsverhältnisse aller Branchen unter einer Woche gelten für die Prüfung der Antragsberechtigung der Neustarthilfe 2022 unter bestimmten Bedingungen (vergleiche 2.3) als selbständige Tätigkeit. Die sich aus diesen Tätigkeiten ergebenden Einkünfte werden entsprechend bei der Bestimmung des Haupterwerbs berücksichtigt. Welche Umsätze beziehungsweise Einnahmen bei der Berechnung der Neustarthilfe Plus zugrunde gelegt werden, ergibt sich aus 3.5, 3.6 und 3.7. Wichtige Hinweise:
- Es ist nur ein Antrag auf Neustarthilfe 2022 pro Förderzeitraum möglich! Wenn Sie einen Antrag als natürliche Person gestellt beziehungsweise Neustarthilfe 2022 in Anspruch genommen haben, kann die Kapitalgesellschaft, deren Gesellschafterin oder Gesellschafter Sie sind, beziehungsweise die Genossenschaft, deren Mitglied Sie sind, grundsätzlich keinen Antrag auf Neustarthilfe 2022 für den gleichen Förderzeitraum stellen und umgekehrt. Ausnahme (vgl. Ziffer 5.1): Wenn Sie als natürliche Person weniger als 25 Prozent der Geschäftsanteile an einer Kapitalgesellschaft halten, können sowohl Sie als natürliche Person als auch die Kapitalgesellschaft weiterhin einen Antrag auf Neustarthilfe 2022 stellen. - Ein Antrag und die Inanspruchnahme von Überbrückungshilfe IV schließt grundsätzlich einen Antrag und die Inanspruchnahme von Neustarthilfe 2022 aus und umgekehrt. Ausnahme (vgl. Ziffer 5.1): Hat eine Kapitalgesellschaft, an der Sie als natürliche Person weniger als 25 Prozent der Geschäftsanteile halten, bereits Überbrückungshilfe IV beantragt oder in Anspruch genommen, können Sie als natürliche Person weiterhin einen Antrag auf Neustarthilfe 2022 stellen. - Wenn Sie Mitglied einer Genossenschaft und gleichzeitig Gesellschafterin oder Gesellschafter einer Kapitalgesellschaft sind, können Sie für den gleichen Förderzeitraum nur entweder im Antrag auf Neustarthilfe 2022 der Kapitalgesellschaft oder im Antrag der Genossenschaft, aber nicht in beiden Anträgen berücksichtigt werden. - Verhältnis zu den Überbrückungshilfen III und Überbrückungshilfen III Plus, einschließlich Neustarthilfe und Neustarthilfe Plus: Ein Antrag und die Inanspruchnahme von Überbrückungshilfe III (Plus), einschließlich der Neustarthilfe (Plus) mit Förderzeitraum Januar bis Juni 2021 (Juli bis Dezember 2021), schließt einen Antrag und die Inanspruchnahme von Neustarthilfe 2022 nicht aus und umgekehrt. - Wahlrecht: Den Antragstellenden wird ein Wahlrecht zwischen der Neustarthilfe 2022 und der Überbrückungshilfe IV eingeräumt, das bis zum 15. Juni 2022 ausgeübt werden kann. Sie können somit von der Neustarthilfe 2022 zur Überbrückungshilfe IV wechseln und umgekehrt. Einzelheiten zum Vorgehen siehe Ziffer 7.

### Input:
Wer ist antragsberechtigt?"""
        )
    )


def test_question_wim():
    log.debug(
        t2tManager.generate(
            """Erzeuge 2 Varianten folgender Frage, die für Embedding-Modelle (Bi-Encoder) geeignet sind: Wo muss ich meine Reisekosten einreichen?"""
        )
    )
